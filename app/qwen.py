"""Standalone Qwen3 text generation CLI (CPU).

Usage:
    python -m app.qwen --prompt "Hello, who are you?"
"""

import argparse
import time
from pathlib import Path

from .config import Settings
from .services import InvalidInputError
from .services.qwen import QwenService


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Generate a text response with Qwen3 on CPU.")
    parser.add_argument("--prompt", required=True, help="User message to send to the model")
    parser.add_argument(
        "--model", default=settings.qwen_model, help="Hugging Face model ID (default: Qwen/Qwen3-0.6B)"
    )
    parser.add_argument("--system", default=settings.qwen_system_prompt, help="System prompt")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=settings.qwen_max_new_tokens,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument("--output", type=Path, help="Optional path to save the response text")
    args = parser.parse_args()

    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be greater than zero")

    started_at = time.perf_counter()
    try:
        qwen = QwenService(args.model, settings.hf_cache_dir, args.system, args.max_new_tokens)
        response, generated_tokens, generation_seconds = qwen.generate(args.prompt)
    except (RuntimeError, InvalidInputError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Response: {response}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Model load time: {qwen.load_seconds:.2f} seconds")
    print(f"Generation time: {generation_seconds:.2f} seconds")
    print(f"Total time: {time.perf_counter() - started_at:.2f} seconds")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{response}\n", encoding="utf-8")
        print(f"Saved response: {args.output}")


if __name__ == "__main__":
    main()
