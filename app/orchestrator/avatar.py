"""MuseTalk avatar integration for the orchestrator.

The MuseTalk service runs on a separate NVIDIA GPU machine (see musetalk/). The
orchestrator sends it the Kokoro WAV, downloads the finished MP4, deletes it on
the GPU machine and serves its own copy at /api/avatar/videos/<id>.mp4, so the
browser only ever talks to localhost:8000 and never sees GPU-side paths.

Avatar problems never fail a voice turn: render() always returns a status dict:
    disabled     MUSETALK_ENABLED is false
    unavailable  service unreachable, loading, unauthorised or busy preparing
    failed       the service accepted the job but generation/download failed
    ready        video_url points at a playable MP4 (Kokoro audio muxed in)
"""

import io
import logging
import re
import time
import uuid
import wave
from pathlib import Path

import httpx

from ..config import Settings

logger = logging.getLogger("stella.avatar")

VIDEO_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}\.mp4$")
IDLE_IMAGE_NAMES = ("idle.png", "idle.jpg", "idle.jpeg", "idle.webp")
# Same rules as the MuseTalk service (musetalk/service/engine.py).
AVATAR_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
REFERENCE_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".png", ".jpg", ".jpeg")
HEALTH_TIMEOUT_SECONDS = 5.0


class AvatarUnavailable(RuntimeError):
    pass


class MuseTalkClient:
    def __init__(self, settings: Settings):
        self.enabled = settings.musetalk_enabled
        self.base_url = settings.musetalk_url
        headers = {"Authorization": f"Bearer {settings.musetalk_api_key}"} if settings.musetalk_api_key else {}
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(settings.musetalk_timeout_seconds, connect=5.0),
        )

    def health(self) -> dict:
        if not self.enabled:
            return {"service": "musetalk", "status": "disabled"}
        try:
            response = self._client.get("/health", timeout=HEALTH_TIMEOUT_SECONDS)
        except httpx.HTTPError as error:
            return {"service": "musetalk", "status": "unreachable", "url": self.base_url, "error": str(error)}
        if response.status_code == 401:
            return {"service": "musetalk", "status": "unauthorized", "error": "MUSETALK_API_KEY does not match the GPU service."}
        try:
            return response.json()
        except ValueError:
            return {"service": "musetalk", "status": "error", "error": f"Unexpected /health response ({response.status_code})."}

    def generate(self, wav: bytes, avatar_id: str) -> tuple[bytes, dict]:
        """Render on the GPU service, download the MP4, then delete it there."""
        try:
            response = self._client.post(
                "/generate",
                files={"audio": ("reply.wav", wav, "audio/wav")},
                data={"avatar_id": avatar_id},
            )
        except httpx.TimeoutException as error:
            raise RuntimeError("MuseTalk timed out (raise MUSETALK_TIMEOUT for long replies).") from error
        except httpx.TransportError as error:
            raise AvatarUnavailable(f"MuseTalk is unreachable: {error}") from error
        if response.status_code in (401, 503):
            raise AvatarUnavailable(_detail(response))
        if response.status_code != 200:
            raise RuntimeError(f"MuseTalk error {response.status_code}: {_detail(response)}")

        info = response.json()
        video_url = info.get("video_url", "")
        if not re.fullmatch(r"/videos/[0-9a-f]{32}\.mp4", video_url):
            raise RuntimeError("MuseTalk returned an unexpected video URL.")
        try:
            video = self._client.get(video_url)
        except httpx.HTTPError as error:
            raise RuntimeError(f"Could not download the MuseTalk video: {error}") from error
        if video.status_code != 200 or not video.content:
            raise RuntimeError(f"Could not download the MuseTalk video ({video.status_code}).")
        try:
            self._client.delete(video_url, timeout=HEALTH_TIMEOUT_SECONDS)
        except httpx.HTTPError:
            logger.warning("Could not delete %s on the MuseTalk service; its TTL cleanup will remove it.", video_url)
        return video.content, info

    def close(self) -> None:
        self._client.close()


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail") or response.text[:300])
    except ValueError:
        return response.text[:300] or response.reason_phrase


class AvatarStore:
    """Short-lived local copies of generated videos, served to the browser."""

    def __init__(self, directory: Path, ttl_minutes: float):
        self.directory = directory
        self.ttl_seconds = ttl_minutes * 60
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, video: bytes) -> str:
        self.cleanup()
        name = f"{uuid.uuid4().hex}.mp4"
        (self.directory / name).write_bytes(video)
        return name

    def path(self, name: str) -> Path | None:
        path = self.directory / name
        return path if VIDEO_NAME_PATTERN.match(name) and path.is_file() else None

    def cleanup(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        for path in self.directory.glob("*.mp4"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass


def idle_image(settings: Settings) -> Path | None:
    folder = settings.avatars_dir / settings.musetalk_avatar_id
    for name in IDLE_IMAGE_NAMES:
        if (folder / name).is_file():
            return folder / name
    return None


def list_avatars(settings: Settings) -> list[dict]:
    """Avatar folders under AVATARS_DIR and the reference file MuseTalk would use for each."""
    found = []
    if settings.avatars_dir.is_dir():
        for folder in sorted(settings.avatars_dir.iterdir()):
            if not (folder.is_dir() and AVATAR_ID_PATTERN.match(folder.name)):
                continue
            reference = next((f"reference{ext}" for ext in REFERENCE_EXTENSIONS if (folder / f"reference{ext}").is_file()), None)
            found.append({"id": folder.name, "reference": reference, "default": folder.name == settings.musetalk_avatar_id})
    return found


def wav_duration(data: bytes) -> float:
    """Duration of a PCM WAV (what MuseTalk accepts). Raises ValueError otherwise."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            frames, rate = wav_file.getnframes(), wav_file.getframerate()
    except (wave.Error, EOFError) as error:
        raise ValueError(f"MuseTalk needs a PCM WAV file ({error}).") from error
    if rate <= 0 or frames / rate <= 0.1:
        raise ValueError("The WAV file is empty or shorter than 0.1 seconds.")
    return frames / rate


def render(client: MuseTalkClient, store: AvatarStore, settings: Settings, wav: bytes, avatar_id: str | None = None) -> dict:
    """Generate the avatar video for a reply. Never raises.

    avatar_id defaults to MUSETALK_AVATAR_ID (conversation turns); the AI Lab may pick another.
    """
    if not settings.musetalk_enabled:
        return {"status": "disabled"}
    started_at = time.perf_counter()
    avatar_id = avatar_id or settings.musetalk_avatar_id
    avatar = {"avatar_id": avatar_id}

    health = client.health()
    if health.get("status") != "ready":
        reason = health.get("error") or f"MuseTalk service is {health.get('status')}"
        return {**avatar, "status": "unavailable", "error": reason,
                "timings": {"avatar_seconds": round(time.perf_counter() - started_at, 3)}}
    try:
        video, info = client.generate(wav, avatar_id)
    except AvatarUnavailable as error:
        return {**avatar, "status": "unavailable", "error": str(error),
                "timings": {"avatar_seconds": round(time.perf_counter() - started_at, 3)}}
    except Exception as error:  # noqa: BLE001 - reported to the browser, voice continues
        logger.warning("Avatar generation failed: %s", error)
        return {**avatar, "status": "failed", "error": str(error),
                "timings": {"avatar_seconds": round(time.perf_counter() - started_at, 3)}}

    name = store.save(video)
    service_timings = info.get("timings", {})
    return {
        **avatar,
        "status": "ready",
        "video_url": f"/api/avatar/videos/{name}",
        "video_bytes": len(video),
        "timings": {
            "musetalk_render_seconds": service_timings.get("render_seconds"),
            "musetalk_service_seconds": service_timings.get("total_seconds"),
            "avatar_seconds": round(time.perf_counter() - started_at, 3),  # incl. upload + download
        },
    }
