"""Orchestrator API: `uvicorn app.orchestrator.main:app --port 8000`.

The only service published to the host. It loads no models; it forwards work to
the whisper, qwen and kokoro containers. It also serves the browser voice UI at
`/` (static files in ./static). File-based endpoints store generated audio under
output/api/; the browser conversation endpoint returns audio inline instead.
"""

import base64
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter, ValidationError

from ..config import Settings
from ..schemas import ChatMessage, GenerateRequest, SynthesizeRequest
from ..servers.common import saved_upload
from ..services import InvalidInputError, ServiceUnavailableError
from ..services.voice import process_voice
from . import avatar
from .clients import KokoroClient, QwenClient, WhisperClient

AUDIO_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+\.wav$")
ERROR_STATUS = {"invalid_input": 422, "unavailable": 503, "failed": 502}
STATIC_DIR = Path(__file__).parent / "static"
HISTORY_ADAPTER = TypeAdapter(list[ChatMessage])


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    timeout = settings.service_timeout_seconds
    app.state.settings = settings
    app.state.audio_dir = settings.output_dir / "api"
    app.state.audio_dir.mkdir(parents=True, exist_ok=True)
    app.state.whisper = WhisperClient("whisper", settings.whisper_url, timeout)
    app.state.qwen = QwenClient("qwen", settings.qwen_url, timeout)
    app.state.kokoro = KokoroClient("kokoro", settings.kokoro_url, timeout)
    # One conversation turn at a time: the CPU models would only queue anyway,
    # and overlapping turns would interleave conversation history.
    app.state.turn_lock = threading.Lock()
    # MuseTalk avatar (remote GPU service); disabled unless MUSETALK_ENABLED=true.
    app.state.musetalk = avatar.MuseTalkClient(settings)
    app.state.avatar_store = avatar.AvatarStore(settings.output_dir / "avatar_videos", settings.avatar_video_ttl_minutes)
    yield
    for client in (app.state.whisper, app.state.qwen, app.state.kokoro, app.state.musetalk):
        client.close()


app = FastAPI(title="Stella Voice Orchestrator", version="0.3.0", lifespan=lifespan)


def _call(fn, *args):
    """Run a client call and map service errors to HTTP status codes."""
    try:
        return fn(*args)
    except InvalidInputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ServiceUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


def _save_wav(request: Request, prefix: str, wav: bytes) -> str:
    name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
    (request.app.state.audio_dir / name).write_bytes(wav)
    return name


def _audio_fields(name: str) -> dict:
    return {"audio_url": f"/api/audio/{name}", "audio_file": f"output/api/{name}"}


def _round(timings: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 3) for key, value in timings.items()}


def _render_avatar(state, wav: bytes, wanted: bool) -> dict:
    if not wanted:
        return {"status": "off"}
    return avatar.render(state.musetalk, state.avatar_store, state.settings, wav)


@app.get("/health")
def health(request: Request):
    """Orchestrator liveness plus each service's readiness (checked in parallel).

    "status" covers the voice services only; the optional MuseTalk avatar is
    reported separately so a missing GPU machine never makes the voice app "degraded".
    """
    state = request.app.state
    clients = (state.whisper, state.qwen, state.kokoro)
    with ThreadPoolExecutor(max_workers=len(clients) + 1) as pool:
        avatar_future = pool.submit(state.musetalk.health)
        services = dict(zip((c.name for c in clients), pool.map(lambda c: c.health(), clients)))
        avatar_health = avatar_future.result()
    all_ready = all(info.get("status") == "ready" for info in services.values())
    return {
        "status": "ok" if all_ready else "degraded",
        "orchestrator": "ok",
        "services": services,
        "avatar": avatar_health,
    }


@app.post("/api/whisper/transcribe")
def whisper_transcribe(request: Request, file: UploadFile = File(...)):
    started_at = time.perf_counter()
    with saved_upload(file, request.app.state.settings.max_upload_mb) as audio_path:
        transcript, seconds = _call(request.app.state.whisper.transcribe, audio_path)
    return {
        "transcript": transcript,
        "timings": _round({"transcription_seconds": seconds, "total_seconds": time.perf_counter() - started_at}),
    }


@app.post("/api/qwen/generate")
def qwen_generate(request: Request, body: GenerateRequest):
    started_at = time.perf_counter()
    history = [message.model_dump() for message in body.history] if body.history else None
    response, tokens, seconds = _call(
        request.app.state.qwen.generate, body.prompt, body.system_prompt, body.max_new_tokens, history
    )
    return {
        "response": response,
        "generated_tokens": tokens,
        "timings": _round({"generation_seconds": seconds, "total_seconds": time.perf_counter() - started_at}),
    }


@app.post("/api/kokoro/synthesize")
def kokoro_synthesize(request: Request, body: SynthesizeRequest):
    started_at = time.perf_counter()
    wav, duration, seconds = _call(request.app.state.kokoro.synthesize_wav, body.text, body.voice, body.speed)
    name = _save_wav(request, "kokoro", wav)
    return {
        **_audio_fields(name),
        "audio_duration_seconds": round(duration, 3),
        "timings": _round({"synthesis_seconds": seconds, "total_seconds": time.perf_counter() - started_at}),
    }


@app.post("/api/voice/process")
def voice_process(
    request: Request,
    file: UploadFile = File(...),
    voice: str | None = Form(default=None),
    speed: float | None = Form(default=None, ge=0.5, le=2.0),
    max_new_tokens: int | None = Form(default=None, ge=1, le=2048),
    avatar_video: bool = Form(default=True, alias="avatar"),
):
    """Audio -> Whisper -> Qwen3 -> Kokoro (-> MuseTalk when enabled). Stage timings are measured
    inside each service; total_seconds is end-to-end wall time in the orchestrator."""
    started_at = time.perf_counter()
    state = request.app.state
    # Fail fast before spending time on Whisper if a later stage is down.
    for client in (state.whisper, state.qwen, state.kokoro):
        _call(client.ensure_ready)

    with saved_upload(file, state.settings.max_upload_mb) as audio_path:
        result = process_voice(
            audio_path, state.whisper, state.qwen, state.kokoro,
            max_new_tokens=max_new_tokens, voice=voice, speed=speed,
        )

    body = {"transcript": result.transcript, "response": result.response, "timings": _round(result.timings)}
    if result.failed_stage:
        body.update(failed_stage=result.failed_stage, error=result.error)
        return JSONResponse(body, status_code=ERROR_STATUS[result.error_kind])

    name = _save_wav(request, "voice", result.wav)
    body.update(_audio_fields(name), audio_duration_seconds=round(result.audio_duration_seconds, 3))
    body["avatar"] = _render_avatar(state, result.wav, avatar_video)
    body["timings"]["total_seconds"] = round(time.perf_counter() - started_at, 3)
    return body


@app.post("/api/conversation/turn")
def conversation_turn(
    request: Request,
    file: UploadFile = File(...),
    history: str = Form(default="[]"),
    voice: str | None = Form(default=None),
    max_new_tokens: int | None = Form(default=None, ge=1, le=512),
    avatar_video: bool = Form(default=True, alias="avatar"),
):
    """One browser voice turn: recorded audio (+ session history) -> transcript, reply, WAV
    (+ MuseTalk avatar video when enabled).

    Accepts any format FFmpeg decodes (browser webm/opus, ogg/opus, mp4/aac, wav);
    the whisper container converts it to 16 kHz PCM. The reply audio is returned
    inline as base64 WAV and never written to disk; the upload lives only in a
    temp file that is deleted when the request ends. The avatar video (Kokoro audio
    muxed in) is kept for AVATAR_VIDEO_TTL_MINUTES and served from /api/avatar/videos/.
    An avatar failure never fails the turn: see the "avatar" object in the response.
    """
    started_at = time.perf_counter()
    state = request.app.state
    try:
        # lstrip: tolerate a UTF-8 BOM (e.g. JSON files written by Windows PowerShell 5.1).
        turns = HISTORY_ADAPTER.validate_python(json.loads(history.lstrip("﻿") or "[]"))
    except (ValueError, ValidationError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid history: {error}") from error
    keep = state.settings.conversation_history_messages
    recent = [turn.model_dump() for turn in turns[-keep:]] if keep > 0 else []

    if not state.turn_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another conversation turn is still being processed.")
    try:
        for client in (state.whisper, state.qwen, state.kokoro):
            _call(client.ensure_ready)
        with saved_upload(file, state.settings.max_upload_mb) as audio_path:
            result = process_voice(
                audio_path, state.whisper, state.qwen, state.kokoro,
                max_new_tokens=max_new_tokens or state.settings.conversation_max_new_tokens,
                voice=voice, history=recent,
            )
        # Still inside the lock: one GPU render at a time, in turn order.
        avatar_result = None if result.failed_stage else _render_avatar(state, result.wav, avatar_video)
    finally:
        state.turn_lock.release()

    body = {
        "transcript": result.transcript,
        "response": result.response,
        "history_messages_used": len(recent),
        "timings": _round(result.timings),
    }
    if result.failed_stage:
        body.update(failed_stage=result.failed_stage, error=result.error, no_speech=result.no_speech)
        return JSONResponse(body, status_code=ERROR_STATUS[result.error_kind])

    body["timings"]["voice_seconds"] = body["timings"].pop("total_seconds")
    body["timings"]["total_seconds"] = round(time.perf_counter() - started_at, 3)
    body.update(
        audio_mime="audio/wav",
        audio_base64=base64.b64encode(result.wav).decode("ascii"),
        audio_duration_seconds=round(result.audio_duration_seconds, 3),
        avatar=avatar_result,
    )
    return body


@app.get("/api/avatar/status")
def avatar_status(request: Request):
    """What the page needs at load time: is the avatar on, is the GPU service ready, is there an idle image."""
    settings = request.app.state.settings
    return {
        "enabled": settings.musetalk_enabled,
        "avatar_id": settings.musetalk_avatar_id,
        "service": request.app.state.musetalk.health(),
        "idle_image_url": "/api/avatar/idle-image" if avatar.idle_image(settings) else None,
    }


@app.get("/api/avatar/avatars")
def avatar_list(request: Request):
    """AI Lab: avatar folders (avatars/<id>/) and their MuseTalk reference file."""
    settings = request.app.state.settings
    return {"default_avatar": settings.musetalk_avatar_id, "avatars": avatar.list_avatars(settings)}


AVATAR_HTTP_STATUS = {"ready": 200, "disabled": 503, "unavailable": 503, "failed": 502}


@app.post("/api/avatar/generate")
def avatar_generate(request: Request, file: UploadFile = File(...), avatar_id: str | None = Form(default=None)):
    """AI Lab: PCM WAV -> MuseTalk only (Whisper, Qwen3 and Kokoro are not called).

    Uses the same render path as conversation turns; the MP4 is served from
    /api/avatar/videos/. Status codes: 200 ready, 503 disabled/unavailable,
    502 failed, 413 too large, 422 bad WAV or avatar id.
    """
    state = request.app.state
    settings = state.settings
    if not settings.musetalk_enabled:
        return JSONResponse(
            {"status": "disabled", "error": "MuseTalk is disabled (MUSETALK_ENABLED=false). It needs the GPU host; see musetalk/README.md."},
            status_code=503,
        )
    avatar_id = (avatar_id or "").strip().lower() or None
    if avatar_id and not avatar.AVATAR_ID_PATTERN.match(avatar_id):
        raise HTTPException(status_code=422, detail=f"Invalid avatar id '{avatar_id}'.")
    limit = settings.max_upload_mb * 1024 * 1024
    wav = file.file.read(limit + 1)
    if len(wav) > limit:
        raise HTTPException(status_code=413, detail=f"Audio exceeds {settings.max_upload_mb} MB.")
    try:
        duration = avatar.wav_duration(wav)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    result = avatar.render(state.musetalk, state.avatar_store, settings, wav, avatar_id)
    result["audio_duration_seconds"] = round(duration, 3)
    return JSONResponse(result, status_code=AVATAR_HTTP_STATUS.get(result["status"], 502))


@app.get("/api/avatar/idle-image")
def avatar_idle_image(request: Request):
    path = avatar.idle_image(request.app.state.settings)
    if path is None:
        raise HTTPException(status_code=404, detail="No idle image configured for this avatar.")
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


@app.get("/api/avatar/videos/{filename}")
def avatar_video(request: Request, filename: str):
    """Generated avatar MP4 (H.264 + AAC). FileResponse supports HTTP Range requests,
    which browsers use for <video> seeking and Safari requires for playback."""
    path = request.app.state.avatar_store.path(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Video not found or expired.")
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/audio/{filename}")
def get_audio(request: Request, filename: str):
    path: Path = request.app.state.audio_dir / filename
    if not AUDIO_NAME_PATTERN.match(filename) or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(path, media_type="audio/wav", filename=filename)
