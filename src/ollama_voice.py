import argparse
import os
import sys
import tempfile
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


def speak_text(tts: TTS, text: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        # Generate speech audio and play it back.
        tts.tts_to_file(text=text, file_path=str(tmp_path))
        data, rate = sf.read(tmp_path, dtype="float32")
        sd.play(data, rate)
        sd.wait()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def interactive_loop(base_url: str, model: str, tts_model: str, device: str | None) -> None:
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
            speak_text(tts, reply)
        except Exception as exc:  # noqa: BLE001 keep loop alive
            print(f"[error] TTS playback failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Ollama using Coqui TTS playback.")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "127.0.0.1"), help="Ollama host or IP")
    parser.add_argument("--port", default=os.environ.get("OLLAMA_PORT", "11434"), help="Ollama port")
    parser.add_argument("--model", default="gurubot/girl", help="Ollama model name")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    try:
        interactive_loop(base_url, args.model, args.tts_model, args.audio_device)
    except KeyboardInterrupt:
        print("\nInterrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
