import argparse
import json
import os
import queue
import re
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import sounddevice as sd
try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency for GPU detection
    ort = None  # type: ignore[assignment]

try:
    import ctranslate2  # type: ignore
except Exception:  # pragma: no cover - optional dependency for GPU detection
    ctranslate2 = None  # type: ignore[assignment]

try:
    from piper import PiperVoice
    from piper.config import PiperConfig
    from piper.download import ensure_voice_exists, find_voice, get_voices
except Exception:  # pragma: no cover - optional runtime dependency
    PiperVoice = None  # type: ignore[assignment]
    PiperConfig = None  # type: ignore[assignment]
    ensure_voice_exists = None  # type: ignore[assignment]
    find_voice = None  # type: ignore[assignment]
    get_voices = None  # type: ignore[assignment]

STOP_SENTINEL = object()
QUEUE_KIND_TEXT = "text"
QUEUE_KIND_VOICE = "voice"

DEFAULT_PIPER_VOICE = "en_US-lessac-low"
PIPER_CACHE_DIR = Path.home() / ".cache" / "ollama-voice-chat" / "piper"


def gpu_is_available() -> bool:
    try:
        if ort is not None and "CUDAExecutionProvider" in set(ort.get_available_providers()):
            return True
    except Exception:
        pass
    try:
        if ctranslate2 is not None and ctranslate2.get_cuda_device_count() > 0:
            return True
    except Exception:
        pass
    return False


def parse_device_string(device: str) -> tuple[str, int | None]:
    text = device.strip().lower()
    if not text:
        return text, None
    if ":" in text:
        base, index_text = text.split(":", 1)
        try:
            return base, int(index_text)
        except ValueError:
            return base, None
    return text, None


def ensure_piper_dependencies() -> None:
    if PiperVoice is None or PiperConfig is None:
        raise RuntimeError("piper-tts package is required for speech playback; reinstall optional dependencies")
    if ort is None:
        raise RuntimeError("onnxruntime is required for speech playback; reinstall optional dependencies")


def _infer_config_path(model_path: Path) -> Path:
    candidates = [
        model_path.with_suffix(model_path.suffix + ".json"),
        model_path.with_suffix(".json"),
        model_path.parent / f"{model_path.name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate Piper config file for {model_path}.")


def load_piper_voice(
    model_name: str,
    config_path_override: str | None,
    download_dir: Path,
    use_cuda: bool,
    cuda_device_index: int,
) -> PiperVoice:
    ensure_piper_dependencies()
    download_dir.mkdir(parents=True, exist_ok=True)

    config_path: Path | None = Path(config_path_override) if config_path_override else None
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(f"Piper config not found at {config_path}.")
    model_path_candidate = Path(model_name)

    if model_path_candidate.exists():
        model_path = model_path_candidate
        if config_path is None:
            config_path = _infer_config_path(model_path)
    else:
        if ensure_voice_exists is None or find_voice is None or get_voices is None:
            raise RuntimeError(
                "Unable to download Piper voice automatically; specify --tts-model with a local .onnx path."
            )
        voices_info = get_voices(download_dir, update_voices=False)
        ensure_voice_exists(model_name, [download_dir], download_dir, voices_info)
        model_path, config_path = find_voice(model_name, [download_dir])

    if config_path is None:
        raise FileNotFoundError(f"Missing Piper config for {model_name}.")

    with open(config_path, "r", encoding="utf-8") as handle:
        config_dict = json.load(handle)

    session_options = ort.SessionOptions()
    if use_cuda:
        providers: list[object] = [
            (
                "CUDAExecutionProvider",
                {"device_id": str(cuda_device_index), "cudnn_conv_algo_search": "HEURISTIC"},
            ),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)
    config = PiperConfig.from_dict(config_dict)
    return PiperVoice(session=session, config=config)


def _ensure_rocm_stub(exc: FileNotFoundError) -> bool:
    created_any = False
    for match in re.finditer(r"'([^']*_rocm_sdk[^']*)'", str(exc)):
        target = Path(match.group(1))
        try:
            (target / "bin").mkdir(parents=True, exist_ok=True)
            created_any = True
        except Exception:
            continue
    return created_any


try:
    from faster_whisper import WhisperModel
except FileNotFoundError as exc:
    if _ensure_rocm_stub(exc):
        from faster_whisper import WhisperModel
    else:  # pragma: no cover - propagate unexpected issues
        raise
except ImportError:  # pragma: no cover - optional runtime dependency
    WhisperModel = None  # type: ignore[assignment]

try:
    import keyboard
except ImportError:  # pragma: no cover - optional runtime dependency
    keyboard = None  # type: ignore[assignment]


def fetch_reply(base_url: str, model: str, prompt: str) -> str:
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    reply = payload.get("response", "").strip()
    if not reply:
        raise RuntimeError("Empty response from Ollama")
    return reply


def iter_speech_chunks(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    buffer = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        if len(sentence) <= max_chars:
            buffer = sentence
            continue
        for wrapped in textwrap.wrap(sentence, max_chars, break_long_words=False, break_on_hyphens=False):
            chunks.append(wrapped)
        buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def speak_text(tts_voice: PiperVoice, text: str, chunk_chars: int) -> None:
    sample_rate = int(tts_voice.config.sample_rate)
    for chunk in iter_speech_chunks(text, chunk_chars):
        audio_segments: list[np.ndarray] = []
        for raw_bytes in tts_voice.synthesize_stream_raw(chunk, sentence_silence=0.05):
            segment = np.frombuffer(raw_bytes, dtype="<i2").astype("float32") / 32768.0
            if segment.size:
                audio_segments.append(segment)
        if not audio_segments:
            continue
        audio = np.concatenate(audio_segments)
        sd.play(audio, sample_rate)
        sd.wait()


def ensure_keyboard_available() -> None:
    if keyboard is None:
        raise RuntimeError("keyboard package is required for push-to-talk; install optional dependencies")


def ensure_stt_available() -> type[Any]:
    if WhisperModel is None:
        raise RuntimeError("faster-whisper package is required for speech recognition; install optional dependencies")
    return WhisperModel  # type: ignore[return-value]


def create_stt_model(
    model_id: str,
    device: str,
    compute_type: str,
    device_index: int | None,
) -> Any:
    model_cls = ensure_stt_available()
    kwargs: dict[str, Any] = {}
    if device.startswith("cuda") and device_index is not None:
        kwargs["device_index"] = device_index
    return model_cls(model_id, device=device, compute_type=compute_type, **kwargs)


def record_voice_session(
    push_to_talk_key: str,
    sample_rate: int,
    block_size: int,
    mic_device: str | int | None,
) -> np.ndarray | None:
    ensure_keyboard_available()
    print(f"Hold '{push_to_talk_key}' to talk, release to send. Press Ctrl+C to cancel.")
    keyboard.wait(push_to_talk_key)
    frames: list[np.ndarray] = []
    start_time = time.perf_counter()
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_size,
        device=mic_device,
    ) as stream:
        while keyboard.is_pressed(push_to_talk_key):
            data, overflowed = stream.read(block_size)
            if overflowed:
                print("[warn] Audio buffer overflowed; transcription quality may degrade")
            frames.append(np.copy(data))
    duration = time.perf_counter() - start_time
    if duration < 0.25 or not frames:
        print("[info] Ignoring very short capture; hold the key longer to speak")
        return None
    audio = np.concatenate(frames, axis=0).flatten()
    return audio


def transcribe_audio(
    stt_model: Any,
    audio: np.ndarray,
    sample_rate: int,
    beam_size: int,
) -> str:
    target_rate = getattr(getattr(stt_model, "feature_extractor", None), "sampling_rate", sample_rate)
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive for transcription")
    audio = audio.astype("float32", copy=False)
    if int(target_rate) != int(sample_rate):
        duration = audio.shape[0] / sample_rate
        target_length = max(1, int(duration * target_rate))
        source_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        audio = np.interp(target_positions, source_positions, audio).astype("float32")
    segments, _ = stt_model.transcribe(
        audio,
        beam_size=beam_size,
        vad_filter=True,
    )
    transcript_parts: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            transcript_parts.append(text)
    return " ".join(transcript_parts).strip()


def start_text_input_worker(
    prompt_queue: "queue.Queue[object]",
    prompt_label: str,
    stop_event: threading.Event,
) -> threading.Thread:
    def worker() -> None:
        while not stop_event.is_set():
            try:
                line = input(prompt_label)
            except EOFError:
                prompt_queue.put(STOP_SENTINEL)
                return
            except KeyboardInterrupt:
                prompt_queue.put(STOP_SENTINEL)
                return
            prompt_queue.put((QUEUE_KIND_TEXT, line))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def start_voice_input_worker(
    prompt_queue: "queue.Queue[object]",
    stop_event: threading.Event,
    push_to_talk_key: str,
    mic_device: str | int | None,
    mic_sample_rate: int,
    voice_block_size: int,
    stt_model: Any,
    stt_beam_size: int,
) -> threading.Thread:
    def worker() -> None:
        while not stop_event.is_set():
            try:
                audio = record_voice_session(
                    push_to_talk_key,
                    mic_sample_rate,
                    voice_block_size,
                    mic_device,
                )
            except KeyboardInterrupt:
                prompt_queue.put(STOP_SENTINEL)
                return
            except Exception as exc:  # noqa: BLE001 keep thread alive
                if stop_event.is_set():
                    return
                print(f"[error] Voice capture failed: {exc}")
                continue
            if audio is None:
                continue
            try:
                prompt = transcribe_audio(stt_model, audio, mic_sample_rate, stt_beam_size)
            except Exception as exc:  # noqa: BLE001 keep thread alive
                if stop_event.is_set():
                    return
                print(f"[error] Transcription failed: {exc}")
                continue
            if not prompt:
                print("[info] Did not detect any speech")
                continue
            prompt_queue.put((QUEUE_KIND_VOICE, prompt))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def interactive_loop(
    base_url: str,
    model: str,
    tts_voice: PiperVoice,
    device: str | None,
    chunk_chars: int,
    push_to_talk: bool,
    push_to_talk_key: str,
    mic_device: str | int | None,
    mic_sample_rate: int,
    stt_model: Any | None,
    stt_beam_size: int,
    voice_block_size: int,
) -> None:
    current_output = sd.default.device
    if device is not None:
        if isinstance(current_output, (tuple, list)) and current_output:
            sd.default.device = (current_output[0], device)
        else:
            sd.default.device = device
    if push_to_talk and stt_model is None:
        raise RuntimeError("Speech recognition model unavailable; cannot enable push-to-talk")
    if not push_to_talk:
        while True:
            try:
                prompt = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                continue
            try:
                reply = fetch_reply(base_url, model, prompt)
            except Exception as exc:  # noqa: BLE001 keep loop alive
                print(f"[error] Ollama request failed: {exc}")
                continue
            print(f"Bot: {reply}")
            try:
                speak_text(tts_voice, reply, chunk_chars)
            except Exception as exc:  # noqa: BLE001 keep loop alive
                print(f"[error] TTS playback failed: {exc}")
        return

    print(
        f"Push-to-talk enabled. Hold '{push_to_talk_key}' to speak at any time, or type messages at the prompt."
    )
    prompt_queue: queue.Queue[object] = queue.Queue()
    stop_event = threading.Event()
    text_thread = start_text_input_worker(prompt_queue, "You (type): ", stop_event)
    voice_thread = start_voice_input_worker(
        prompt_queue,
        stop_event,
        push_to_talk_key,
        mic_device,
        mic_sample_rate,
        voice_block_size,
        stt_model,
        stt_beam_size,
    )
    worker_threads = [text_thread, voice_thread]
    try:
        while True:
            try:
                item = prompt_queue.get()
            except KeyboardInterrupt:
                print()
                break
            if item is STOP_SENTINEL:
                break
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            kind, raw_prompt = item
            prompt = str(raw_prompt).strip()
            if not prompt:
                continue
            if kind == QUEUE_KIND_VOICE:
                print(f"You (voice): {prompt}")
            try:
                reply = fetch_reply(base_url, model, prompt)
            except Exception as exc:  # noqa: BLE001 keep loop alive
                print(f"[error] Ollama request failed: {exc}")
                continue
            print(f"Bot: {reply}")
            try:
                speak_text(tts_voice, reply, chunk_chars)
            except Exception as exc:  # noqa: BLE001 keep loop alive
                print(f"[error] TTS playback failed: {exc}")
    finally:
        stop_event.set()
        prompt_queue.put(STOP_SENTINEL)
        for worker in worker_threads:
            worker.join(timeout=0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Ollama using Piper TTS playback.")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "127.0.0.1"), help="Ollama host or IP")
    parser.add_argument("--port", default=os.environ.get("OLLAMA_PORT", "11434"), help="Ollama port")
    parser.add_argument("--model", default="deepseek-r1:7b", help="Ollama model name")
    parser.add_argument(
        "--tts-model",
        default=DEFAULT_PIPER_VOICE,
        help="Piper voice identifier or local path to a .onnx model",
    )
    parser.add_argument(
        "--tts-config",
        default=None,
        help="Optional path to Piper voice config (use with local .onnx models)",
    )
    parser.add_argument(
        "--tts-device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Preferred device for TTS synthesis (auto selects GPU when available)",
    )
    parser.add_argument(
        "--tts-gpu-index",
        type=int,
        default=0,
        help="CUDA device index for TTS when GPU acceleration is enabled",
    )
    parser.add_argument(
        "--tts-voices-dir",
        default=None,
        help="Cache directory for downloaded Piper voices (default: %USERPROFILE%/.cache/ollama-voice-chat/piper)",
    )
    parser.add_argument(
        "--audio-device",
        default=None,
        help="Optional sounddevice output identifier (name or index)",
    )
    parser.add_argument(
        "--tts-chunk-chars",
        type=int,
        default=220,
        help="Maximum characters per synthesized audio chunk",
    )
    parser.add_argument(
        "--push-to-talk",
        action="store_true",
        help="Enable microphone capture triggered by holding the push-to-talk key",
    )
    parser.add_argument(
        "--push-to-talk-key",
        default="right shift",
        help="Keyboard key to hold while speaking (uses the 'keyboard' package naming)",
    )
    parser.add_argument(
        "--mic-device",
        default=None,
        help="Optional sounddevice input identifier (name or index)",
    )
    parser.add_argument(
        "--mic-sample-rate",
        type=int,
        default=16000,
        help="Microphone capture sample rate",
    )
    parser.add_argument(
        "--voice-block-size",
        type=int,
        default=2048,
        help="Audio block size for microphone capture",
    )
    parser.add_argument(
        "--stt-model",
        default="base.en",
        help="faster-whisper speech-to-text model identifier or local path",
    )
    parser.add_argument(
        "--stt-device",
        default="auto",
        help="Hardware device for faster-whisper (auto picks cuda when available)",
    )
    parser.add_argument(
        "--stt-gpu-index",
        type=int,
        default=0,
        help="CUDA device index for faster-whisper when running on GPU",
    )
    parser.add_argument(
        "--stt-compute-type",
        default="auto",
        help="Compute precision for faster-whisper (auto floats to float16 on GPU, int8 on CPU)",
    )
    parser.add_argument(
        "--stt-beam-size",
        type=int,
        default=1,
        help="Beam search width for speech recognition",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    gpu_available = gpu_is_available()
    tts_choice = args.tts_device.lower()
    tts_gpu_index = args.tts_gpu_index
    if tts_choice == "gpu":
        if not gpu_available:
            print("[warn] GPU requested for TTS but no CUDA device detected; falling back to CPU")
        tts_use_gpu = gpu_available
    elif tts_choice == "cpu":
        tts_use_gpu = False
    else:  # auto
        tts_use_gpu = gpu_available

    voices_dir = Path(args.tts_voices_dir) if args.tts_voices_dir else PIPER_CACHE_DIR
    try:
        tts_voice = load_piper_voice(args.tts_model, args.tts_config, voices_dir, tts_use_gpu, tts_gpu_index)
    except Exception as exc:  # noqa: BLE001 surface initialization failure
        print(f"[error] Unable to initialize Piper voice '{args.tts_model}': {exc}")
        return 1

    provider_names = tuple(tts_voice.session.get_providers())
    if "CUDAExecutionProvider" in provider_names:
        provider_options = tts_voice.session.get_provider_options().get("CUDAExecutionProvider", {})
        device_id = provider_options.get("device_id", str(tts_gpu_index))
        if tts_use_gpu:
            print(f"[info] Piper TTS initialized on CUDA device {device_id}.")
        else:
            print(f"[info] Piper TTS auto-selected CUDA device {device_id}.")
    elif tts_use_gpu:
        print("[warn] Requested CUDA for TTS but CUDAExecutionProvider is unavailable; using CPU instead.")

    stt_device_arg = str(args.stt_device).strip()
    stt_device_base, stt_device_index_override = parse_device_string(stt_device_arg)

    stt_gpu_index = args.stt_gpu_index
    stt_device: str
    stt_device_index: int | None = None
    if stt_device_base in {"", "auto"}:
        if gpu_available:
            stt_device = "cuda"
            stt_device_index = stt_gpu_index
        else:
            stt_device = "cpu"
    elif stt_device_base in {"gpu", "cuda"}:
        if not gpu_available:
            print("[warn] GPU requested for STT but no CUDA device detected; using CPU instead")
            stt_device = "cpu"
        else:
            stt_device = "cuda"
            if stt_device_index_override is not None:
                stt_device_index = stt_device_index_override
            else:
                stt_device_index = stt_gpu_index
    else:
        stt_device = stt_device_base or stt_device_arg
        stt_device_index = stt_device_index_override

    stt_compute_arg = str(args.stt_compute_type).strip()
    if stt_compute_arg.lower() == "auto":
        stt_compute_type = "float16" if stt_device.startswith("cuda") else "int8_float32"
    else:
        stt_compute_type = stt_compute_arg

    stt_model = None
    if args.push_to_talk:
        ensure_keyboard_available()
        if stt_device.startswith("cuda") and stt_device_index is not None:
            device_label = f"{stt_device}:{stt_device_index}"
        else:
            device_label = stt_device
        print(f"[info] Initializing speech recognition on {device_label} ({stt_compute_type}).")
        try:
            stt_model = create_stt_model(args.stt_model, stt_device, stt_compute_type, stt_device_index)
        except Exception as exc:  # noqa: BLE001 - bubble up in CLI context
            print(f"[error] Unable to initialize speech recognition: {exc}")
            return 1
    try:
        interactive_loop(
            base_url,
            args.model,
            tts_voice,
            args.audio_device,
            args.tts_chunk_chars,
            args.push_to_talk,
            args.push_to_talk_key,
            args.mic_device,
            args.mic_sample_rate,
            stt_model,
            args.stt_beam_size,
            args.voice_block_size,
        )
    except KeyboardInterrupt:
        print("\nInterrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
