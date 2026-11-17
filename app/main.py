"""Standalone Whisper transcription CLI (CPU).

Usage:
    python -m app.main --input samples/input.wav
"""




import argparse
import time
from pathlib import Path

from .config import Settings
from .services import InvalidInputError
from .services.whisper import WhisperService


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Transcribe an audio file with Whisper.")
    parser.add_argument("--input", required=True, type=Path, help="Audio file to transcribe")
    parser.add_argument("--model", default=settings.whisper_model, help="Whisper model name (default: base)")
    parser.add_argument("--output", type=Path, help="Transcript path (default: output/<input>.txt)")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input audio file does not exist: {args.input}")

    started_at = time.perf_counter()
    try:
        whisper = WhisperService(args.model, settings.model_cache_dir)
        transcript, transcription_seconds = whisper.transcribe(args.input)
    except (RuntimeError, InvalidInputError) as error:
        parser.exit(1, f"Error: {error}\n")

    transcript_path = args.output or settings.output_dir / f"{args.input.stem}.txt"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(f"{transcript}\n", encoding="utf-8")

    print(f"Transcription: {transcript}")
    print(f"Model load time: {whisper.load_seconds:.2f} seconds")
    print(f"Transcription time: {transcription_seconds:.2f} seconds")
    print(f"Total time: {time.perf_counter() - started_at:.2f} seconds")
    print(f"Saved transcript: {transcript_path}")


if __name__ == "__main__":
    main()
