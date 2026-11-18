"""Qwen3 text-generation service (CPU, thinking mode disabled)."""

import threading
import time
from pathlib import Path

from . import InvalidInputError, require_module

# Sampling settings recommended by Qwen for non-thinking mode. Shared with the
# orchestrator's vLLM client so both backends answer the same way.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def build_chat_messages(system_prompt: str, history: list[dict] | None, prompt: str) -> list[dict]:
    """System prompt, then earlier turns (oldest first), then the new user prompt.

    history: [{"role": "user"|"assistant", "content": str}, ...]; other roles and
    empty messages are skipped.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history or []:
        if turn.get("role") in ("user", "assistant") and str(turn.get("content", "")).strip():
            messages.append({"role": turn["role"], "content": str(turn["content"]).strip()})
    messages.append({"role": "user", "content": prompt})
    return messages


class QwenService:
    def __init__(self, model_name: str, cache_dir: Path, system_prompt: str, max_new_tokens: int):
        self._torch = require_module("torch")
        transformers = require_module("transformers")
        cache_dir.mkdir(parents=True, exist_ok=True)

        started_at = time.perf_counter()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            dtype=self._torch.float32,
        )
        self.model.to("cpu")
        self.model.eval()
        self.load_seconds = time.perf_counter() - started_at

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        history: list[dict] | None = None,
    ) -> tuple[str, int, float]:
        """Return (response, generated_token_count, seconds).

        history: earlier turns as [{"role": "user"|"assistant", "content": str}, ...],
        oldest first; inserted between the system prompt and the new prompt.
        """
        prompt = prompt.strip()
        if not prompt:
            raise InvalidInputError("Prompt must not be empty.")
        max_new_tokens = max_new_tokens or self.max_new_tokens
        if max_new_tokens < 1:
            raise InvalidInputError("max_new_tokens must be greater than zero.")

        messages = build_chat_messages(system_prompt or self.system_prompt, history, prompt)
        # enable_thinking=False makes the Qwen3 chat template skip the <think> phase.
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt")

        with self._lock, self._torch.inference_mode():
            started_at = time.perf_counter()
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                **SAMPLING,
            )
            seconds = time.perf_counter() - started_at

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response, len(new_tokens), seconds
