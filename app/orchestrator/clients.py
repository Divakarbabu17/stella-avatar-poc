"""HTTP clients for the model services.

They expose the same methods as the in-process services in app/services, so
app.services.voice.process_voice can drive either. Every call first checks the
service's /health and fails fast with ServiceUnavailableError if it is not ready.
"""

from pathlib import Path

import httpx

from ..services import InvalidInputError, ServiceUnavailableError

HEALTH_TIMEOUT_SECONDS = 5.0


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return str(detail or response.text[:300] or response.reason_phrase)


class ServiceClient:
    def __init__(self, name: str, base_url: str, timeout_seconds: float):
        self.name = name
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout_seconds, connect=5.0))

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
        if response.status_code in (400, 413, 422):
            raise InvalidInputError(_detail(response))
        if response.status_code == 503:
            raise ServiceUnavailableError(_detail(response))
        if response.status_code >= 400:
            raise RuntimeError(f"{self.name} service error {response.status_code}: {_detail(response)}")
        return response

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
