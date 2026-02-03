import argparse
import os
import re
import sys
import tempfile
import textwrap
from pathlib import Path

import requests
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS


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


def interactive_loop(
    base_url: str,
    model: str,
    tts_model: str,
    device: str | None,
    chunk_chars: int,
) -> None:
    tts = TTS(model_name=tts_model)
    if device is not None:
        sd.default.device = device
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    try:
        interactive_loop(
            base_url,
            args.model,
            args.tts_model,
            args.audio_device,
            args.tts_chunk_chars,
        )
    except KeyboardInterrupt:
        print("\nInterrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
