import argparse
import os
import queue
import re
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS

STOP_SENTINEL = object()
QUEUE_KIND_TEXT = "text"
QUEUE_KIND_VOICE = "voice"


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


def speak_text(tts: TTS, text: str, chunk_chars: int) -> None:
    for chunk in iter_speech_chunks(text, chunk_chars):
        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            # Generate speech audio in small chunks to reduce latency.
            tts.tts_to_file(text=chunk, file_path=str(tmp_path))
            data, rate = sf.read(tmp_path, dtype="float32")
            sd.play(data, rate)
            sd.wait()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


def ensure_keyboard_available() -> None:
    if keyboard is None:
        raise RuntimeError("keyboard package is required for push-to-talk; install optional dependencies")


def ensure_stt_available() -> type[Any]:
    if WhisperModel is None:
        raise RuntimeError("faster-whisper package is required for speech recognition; install optional dependencies")
    return WhisperModel  # type: ignore[return-value]


def create_stt_model(model_id: str, device: str, compute_type: str) -> Any:
    model_cls = ensure_stt_available()
    return model_cls(model_id, device=device, compute_type=compute_type)


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
    tts_model: str,
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
    tts = TTS(model_name=tts_model)
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
                speak_text(tts, reply, chunk_chars)
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
                speak_text(tts, reply, chunk_chars)
            except Exception as exc:  # noqa: BLE001 keep loop alive
                print(f"[error] TTS playback failed: {exc}")
    finally:
        stop_event.set()
        prompt_queue.put(STOP_SENTINEL)
        for worker in worker_threads:
            worker.join(timeout=0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Ollama using Coqui TTS playback.")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "127.0.0.1"), help="Ollama host or IP")
    parser.add_argument("--port", default=os.environ.get("OLLAMA_PORT", "11434"), help="Ollama port")
    parser.add_argument("--model", default="deepseek-r1:7b", help="Ollama model name")
    parser.add_argument(
        "--tts-model",
        default="tts_models/en/jenny/jenny",
        help="Coqui TTS model identifier",
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
        default="cpu",
        help="Hardware device for faster-whisper (e.g. cpu, cuda)",
    )
    parser.add_argument(
        "--stt-compute-type",
        default="int8_float32",
        help="Compute precision for faster-whisper (e.g. float16, int8_float32)",
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
    stt_model = None
    if args.push_to_talk:
        ensure_keyboard_available()
        try:
            stt_model = create_stt_model(args.stt_model, args.stt_device, args.stt_compute_type)
        except Exception as exc:  # noqa: BLE001 - bubble up in CLI context
            print(f"[error] Unable to initialize speech recognition: {exc}")
            return 1
    try:
        interactive_loop(
            base_url,
            args.model,
            args.tts_model,
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
