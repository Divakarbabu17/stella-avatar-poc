"""Run the CPU voice pipeline from audio to a Qwen3 text response (CLI).

Kokoro is intentionally not part of this CLI; the full Whisper -> Qwen3 -> Kokoro
pipeline is served by POST /api/voice/process.
"""

import argparse
import time
from pathlib import Path

from .config import Settings
from .services.qwen import QwenService
from .services.voice import process_voice
from .services.whisper import WhisperService


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Transcribe audio and generate a Qwen3 response.")
    parser.add_argument("--input", required=True, type=Path, help="Audio file to process")
    parser.add_argument("--whisper-model", default=settings.whisper_model, help="Whisper model name (default: base)")
    parser.add_argument("--qwen-model", default=settings.qwen_model, help="Qwen3 Hugging Face model ID")
    parser.add_argument("--system", default=settings.qwen_system_prompt, help="System prompt for Qwen3")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=settings.qwen_max_new_tokens,
        help="Maximum number of response tokens",
    )
    parser.add_argument("--output", type=Path, help="Report path (default: output/<input>_pipeline.txt)")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input audio file does not exist: {args.input}")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be greater than zero")

    pipeline_started_at = time.perf_counter()
    transcription_seconds = generation_seconds = 0.0
    transcript, response, failed, transcription_failed = "", "", False, False
    try:
        whisper = WhisperService(args.whisper_model, settings.model_cache_dir)
        qwen = QwenService(args.qwen_model, settings.hf_cache_dir, args.system, args.max_new_tokens)
    except Exception as error:
        response = f"Not generated because model loading failed: {error}"
        failed = transcription_failed = True
    else:
        result = process_voice(args.input, whisper, qwen)
        transcript, response = result.transcript, result.response
        transcription_seconds = result.timings.get("transcription_seconds", 0.0)
        generation_seconds = result.timings.get("generation_seconds", 0.0)
        if result.no_speech:
            response = "Not generated because Whisper returned an empty transcript."
        elif result.failed_stage == "transcription":
            response = f"Not generated because transcription failed: {result.error}"
            failed = transcription_failed = True
        elif result.failed_stage == "generation":
            response = f"Qwen3 generation failed: {result.error}"
            failed = True

    total_seconds = time.perf_counter() - pipeline_started_at
    transcript_text = "[Transcription failed]" if transcription_failed else transcript or "[No speech detected]"
    report = (
        f"Transcription:\n{transcript_text}\n\n"
        f"Qwen3 response:\n{response}\n\n"
        f"Transcription time: {transcription_seconds:.2f} seconds\n"
        f"LLM generation time: {generation_seconds:.2f} seconds\n"
        f"Total pipeline time: {total_seconds:.2f} seconds\n"
    )
    report_path = args.output or settings.output_dir / f"{args.input.stem}_pipeline.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Transcription: {transcript_text}")
    print(f"Qwen3 response: {response}")
    print(f"Transcription time: {transcription_seconds:.2f} seconds")
    print(f"LLM generation time: {generation_seconds:.2f} seconds")
    print(f"Total pipeline time: {total_seconds:.2f} seconds")
    print(f"Saved pipeline report: {report_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
