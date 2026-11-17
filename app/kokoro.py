"""Standalone Kokoro text-to-speech CLI (CPU).

Usage:
    python -m app.kokoro --text "Hello! Welcome to our AI assistant."
"""

import argparse
import time
from pathlib import Path

from .config import Settings
from .services import InvalidInputError
from .services.kokoro import SAMPLE_RATE, SUPPORTED_LANG_CODES, KokoroService, to_wav_bytes, validate_voice


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Generate speech from text with Kokoro on CPU.")
    parser.add_argument("--text", required=True, help="Text to speak")
    parser.add_argument(
        "--voice",
        default=settings.kokoro_voice,
        help="Kokoro voice ID, e.g. af_heart, af_bella, am_michael, bf_emma (default: af_heart)",
    )
    parser.add_argument("--speed", type=float, default=settings.kokoro_speed, help="Speed multiplier (0.5-2.0)")
    parser.add_argument("--repo-id", default=settings.kokoro_repo_id, help="Hugging Face model repository")
    parser.add_argument("--output", type=Path, help="WAV path (default: output/kokoro_<voice>.wav)")
    args = parser.parse_args()

    text = args.text.strip()
    if not text:
        parser.error("--text must not be empty")
    try:
        voice = validate_voice(args.voice)
    except InvalidInputError as error:
        parser.error(str(error))
    if not 0.5 <= args.speed <= 2.0:
        parser.error("--speed must be between 0.5 and 2.0")
    output_path = args.output or settings.output_dir / f"kokoro_{voice}.wav"
    if output_path.suffix.lower() != ".wav":
        parser.error("--output must end with .wav")

    started_at = time.perf_counter()
    try:
        # Loading with the requested voice as default also downloads/caches that voice pack.
        kokoro = KokoroService(args.repo_id, settings.hf_cache_dir, voice, args.speed)
        audio, generation_seconds = kokoro.synthesize(text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(to_wav_bytes(audio))
    except (RuntimeError, InvalidInputError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")

    duration_seconds = len(audio) / SAMPLE_RATE
    print(f"Model: {args.repo_id}")
    print(f"Voice: {voice} ({SUPPORTED_LANG_CODES[voice[0]]})")
    print(f"Text: {text}")
    print(f"Model load time (incl. voice): {kokoro.load_seconds:.2f} seconds")
    print(f"Audio generation time: {generation_seconds:.2f} seconds")
    print(f"Audio duration: {duration_seconds:.2f} seconds")
    print(f"Real-time factor: {generation_seconds / duration_seconds:.2f}")
    print(f"Total time: {time.perf_counter() - started_at:.2f} seconds")
    print(f"Saved audio: {output_path}")


if __name__ == "__main__":
    main()
