"""HTTP clients for the model services.

They expose the same methods as the in-process services in app/services, so
app.services.voice.process_voice can drive either. Every call first checks the
service's /health and fails fast with ServiceUnavailableError if it is not ready.
"""

import json
import re
import time
from pathlib import Path

import httpx

from ..services import InvalidInputError, ServiceUnavailableError
from ..services.qwen import SAMPLING, build_chat_messages

HEALTH_TIMEOUT_SECONDS = 5.0
# Defensive: Qwen3 hybrid-thinking models emit <think>...</think> only when thinking is
# on. We turn it off per request; this strips any block that still slips through.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# vLLM Prometheus metrics shown in the System status panel (name -> key). The KV-cache
# gauge was renamed across vLLM versions; both names are accepted.
VLLM_METRICS = {
    "vllm:kv_cache_usage_perc": "kv_cache_usage",
    "vllm:gpu_cache_usage_perc": "kv_cache_usage",
    "vllm:num_requests_running": "requests_running",
    "vllm:num_requests_waiting": "requests_waiting",
}


def _rate(tokens: int, seconds: float | None) -> float | None:
    return round(tokens / seconds, 2) if tokens and seconds and seconds > 0 else None


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return str(detail or response.text[:300] or response.reason_phrase)


class ServiceClient:
    def __init__(self, name: str, base_url: str, timeout_seconds: float, headers: dict[str, str] | None = None):
        self.name = name
        self.base_url = base_url
        self._client = httpx.Client(
            base_url=base_url, timeout=httpx.Timeout(timeout_seconds, connect=5.0), headers=headers or {}
        )

    def health(self) -> dict:
        """The service's own /health body, or a synthetic one if it is unreachable."""
        try:
            response = self._client.get("/health", timeout=HEALTH_TIMEOUT_SECONDS)
            return response.json()
        except (httpx.HTTPError, ValueError) as error:
            return {"service": self.name, "status": "unreachable", "url": self.base_url, "error": str(error)}

    def ensure_ready(self) -> None:
        health = self.health()
        if health.get("status") != "ready":
            reason = health.get("error") or health.get("status")
            raise ServiceUnavailableError(f"{self.name} service is not ready ({reason}).")

    def _post(self, path: str, **kwargs) -> httpx.Response:
        self.ensure_ready()
        try:
            response = self._client.post(path, **kwargs)
        except httpx.TimeoutException as error:
            raise RuntimeError(f"{self.name} service timed out") from error
        except httpx.TransportError as error:
            raise ServiceUnavailableError(f"{self.name} service is unreachable: {error}") from error
        self._raise_for_status(response)
        return response

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code in (400, 413, 422):
            raise InvalidInputError(_detail(response))
        if response.status_code == 503:
            raise ServiceUnavailableError(_detail(response))
        if response.status_code >= 400:
            raise RuntimeError(f"{self.name} service error {response.status_code}: {_detail(response)}")

    def gpu(self) -> dict | None:
        """The service's GET /gpu (nvidia-smi snapshot), or None if it has none or is down."""
        try:
            response = self._client.get("/gpu", timeout=HEALTH_TIMEOUT_SECONDS)
            return response.json() if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    def close(self) -> None:
        self._client.close()


class WhisperClient(ServiceClient):
    def transcribe(self, audio_path: Path) -> tuple[str, float]:
        with open(audio_path, "rb") as audio_file:
            data = self._post("/transcribe", files={"file": (audio_path.name, audio_file)}).json()
        return data["transcript"], data["transcription_seconds"]


class QwenClient(ServiceClient):
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        history: list[dict] | None = None,
    ) -> tuple[str, int, float]:
        body = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_new_tokens": max_new_tokens,
            "history": history or None,
        }
        data = self._post("/generate", json=body).json()
        return data["response"], data["generated_tokens"], data["generation_seconds"]

    def generate_with_metrics(self, prompt, system_prompt=None, max_new_tokens=None, history=None) -> dict:
        """Same as generate() plus speed figures. The transformers service does not
        stream, so time to first token is not available for this backend."""
        response, tokens, seconds = self.generate(prompt, system_prompt, max_new_tokens, history)
        return {
            "response": response, "generated_tokens": tokens, "generation_seconds": seconds,
            "time_to_first_token_seconds": None, "tokens_per_second": _rate(tokens, seconds),
            "decode_tokens_per_second": None, "finish_reason": None, "backend": "transformers",
        }


class VllmQwenClient(ServiceClient):
    """Qwen3 served by vLLM's OpenAI-compatible API (QWEN_BACKEND=vllm).

    Same generate() signature and return value as QwenClient, so the conversation
    flow and the AI Lab work unchanged. The default system prompt and reply-length
    cap, which the transformers service applied itself, are applied here.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        timeout_seconds: float,
        *,
        model: str,
        api_key: str,
        system_prompt: str,
        max_new_tokens: int,
    ):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        super().__init__(name, base_url, timeout_seconds, headers=headers)
        self.model = model
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

    def health(self) -> dict:
        """Ready = vLLM's /health answers 200 AND /v1/models lists QWEN_MODEL.

        vLLM's /health returns 200 with an EMPTY body, so only its status code is
        used. /v1/models (protected by the API key, unlike /health) confirms the
        key and the served model name.
        """
        base = {"service": self.name, "backend": "vllm", "model": self.model, "url": self.base_url}
        try:
            live = self._client.get("/health", timeout=HEALTH_TIMEOUT_SECONDS)
        except httpx.HTTPError as error:
            # vLLM only opens its HTTP port after the model has loaded.
            return {**base, "status": "unreachable", "error": f"{error} (vLLM listens only once the model is loaded)"}
        if live.status_code != 200:
            return {**base, "status": "error", "error": f"vLLM /health returned HTTP {live.status_code}."}
        try:
            listed = self._client.get("/v1/models", timeout=HEALTH_TIMEOUT_SECONDS)
        except httpx.HTTPError as error:
            return {**base, "status": "unreachable", "error": str(error)}
        if listed.status_code == 401:
            return {**base, "status": "unauthorized", "error": "QWEN_API_KEY does not match the vLLM API key."}
        try:
            served = [entry["id"] for entry in listed.json()["data"]]
        except (ValueError, KeyError, TypeError) as error:
            return {**base, "status": "error", "error": f"Unexpected /v1/models response (HTTP {listed.status_code}): {error}"}
        if self.model not in served:
            return {**base, "status": "error", "error": f"vLLM serves {served}, not '{self.model}' (check QWEN_MODEL)."}
        return {**base, "status": "ready", "served_models": served}

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        history: list[dict] | None = None,
    ) -> tuple[str, int, float]:
        result = self.generate_with_metrics(prompt, system_prompt, max_new_tokens, history)
        return result["response"], result["generated_tokens"], result["generation_seconds"]

    def generate_with_metrics(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        """One chat completion, streamed from vLLM so time to first token can be measured.

        The caller still receives the complete reply at once. Times are measured here
        (orchestrator side, over the internal Docker network).
        """
        prompt = prompt.strip()
        if not prompt:
            raise InvalidInputError("Prompt must not be empty.")
        max_new_tokens = max_new_tokens or self.max_new_tokens
        if max_new_tokens < 1:
            raise InvalidInputError("max_new_tokens must be greater than zero.")
        body = {
            "model": self.model,
            "messages": build_chat_messages(system_prompt or self.system_prompt, history, prompt),
            "max_completion_tokens": max_new_tokens,
            **SAMPLING,
            # Qwen3 hybrid models: skip the <think> phase (same as the transformers service).
            # Chat templates without this variable ignore it.
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
            "stream_options": {"include_usage": True},  # final chunk carries the token counts
        }
        self.ensure_ready()
        parts: list[str] = []
        usage: dict | None = None
        finish_reason = None
        first_at = last_at = None
        started_at = time.perf_counter()
        try:
            with self._client.stream("POST", "/v1/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    response.read()
                    self._raise_for_status(response)
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    if chunk.get("error"):
                        raise RuntimeError(f"vLLM stream error: {chunk['error']}")
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            now = time.perf_counter()
                            if first_at is None:
                                first_at = now
                            last_at = now
                            parts.append(text)
                        finish_reason = choice.get("finish_reason") or finish_reason
        except httpx.TimeoutException as error:
            raise RuntimeError(f"{self.name} service timed out") from error
        except httpx.TransportError as error:
            raise ServiceUnavailableError(f"{self.name} service is unreachable: {error}") from error
        seconds = time.perf_counter() - started_at

        reply = THINK_BLOCK.sub("", "".join(parts)).strip()
        if not reply:
            raise RuntimeError(f"vLLM returned an empty reply (finish_reason={finish_reason}).")
        # usage is sent when stream_options.include_usage is honoured; otherwise count chunks.
        tokens = int((usage or {}).get("completion_tokens") or len(parts))
        decode_seconds = (last_at - first_at) if first_at is not None and last_at is not None else None
        return {
            "response": reply,
            "generated_tokens": tokens,
            "generation_seconds": seconds,
            "time_to_first_token_seconds": (first_at - started_at) if first_at is not None else None,
            "tokens_per_second": _rate(tokens, seconds),
            # Steady-state speed after the first token (excludes prompt processing).
            "decode_tokens_per_second": _rate(tokens - 1, decode_seconds) if tokens > 1 else None,
            "finish_reason": finish_reason,
            "backend": "vllm",
        }

    def engine_metrics(self) -> dict | None:
        """KV-cache usage (0-1) and request queue from vLLM's Prometheus /metrics (not key-protected)."""
        try:
            response = self._client.get("/metrics", timeout=HEALTH_TIMEOUT_SECONDS)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        found: dict[str, float] = {}
        for line in response.text.splitlines():
            if not line or line.startswith("#"):
                continue
            name, _, rest = line.partition("{") if "{" in line.split(" ", 1)[0] else line.partition(" ")
            key = VLLM_METRICS.get(name.strip())
            if key is None:
                continue
            value_part = rest.rsplit("}", 1)[-1] if "}" in rest else rest
            try:
                value = float(value_part.split()[0])
            except (IndexError, ValueError):
                continue
            # Usage: highest of any series; queue counts: summed across series.
            found[key] = max(found.get(key, 0.0), value) if key == "kv_cache_usage" else found.get(key, 0.0) + value
        return found or None


class KokoroClient(ServiceClient):
    def synthesize_wav(
        self, text: str, voice: str | None = None, speed: float | None = None
    ) -> tuple[bytes, float, float]:
        response = self._post("/synthesize", json={"text": text, "voice": voice, "speed": speed})
        return (
            response.content,
            float(response.headers["X-Audio-Duration-Seconds"]),
            float(response.headers["X-Synthesis-Seconds"]),
        )
