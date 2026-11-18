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
from collections import deque
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
from .clients import KokoroClient, QwenClient, VllmQwenClient, WhisperClient

AUDIO_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+\.wav$")
ERROR_STATUS = {"invalid_input": 422, "unavailable": 503, "failed": 502}
STATIC_DIR = Path(__file__).parent / "static"
HISTORY_ADAPTER = TypeAdapter(list[ChatMessage])


class PerfLog:
    """Recent processing times per stage (in memory, last N runs; reset on restart).

    Fed by the AI Lab endpoints and by conversation turns; shown in the System status panel.
    """

    def __init__(self, size: int = 20):
        self._size = size
        self._runs: dict[str, deque] = {}
        self._lock = threading.Lock()

    def add(self, stage: str, seconds) -> None:
        if not isinstance(seconds, (int, float)) or seconds < 0:
            return
        with self._lock:
            self._runs.setdefault(stage, deque(maxlen=self._size)).append((time.time(), float(seconds)))

    def summary(self) -> dict:
        with self._lock:
            runs = {stage: list(values) for stage, values in self._runs.items()}
        out = {}
        for stage, values in runs.items():
            seconds = [s for _, s in values]
            out[stage] = {
                "last_seconds": round(seconds[-1], 3), "avg_seconds": round(sum(seconds) / len(seconds), 3),
                "min_seconds": round(min(seconds), 3), "max_seconds": round(max(seconds), 3),
                "count": len(seconds), "last_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(values[-1][0])),
            }
        return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    timeout = settings.service_timeout_seconds
    app.state.settings = settings
    app.state.audio_dir = settings.output_dir / "api"
    app.state.audio_dir.mkdir(parents=True, exist_ok=True)
    app.state.whisper = WhisperClient("whisper", settings.whisper_url, timeout)
    if settings.qwen_backend == "vllm":
        app.state.qwen = VllmQwenClient(
            "qwen", settings.qwen_url, timeout,
            model=settings.qwen_model, api_key=settings.qwen_api_key,
            system_prompt=settings.qwen_system_prompt, max_new_tokens=settings.qwen_max_new_tokens,
        )
    else:
        app.state.qwen = QwenClient("qwen", settings.qwen_url, timeout)
    app.state.kokoro = KokoroClient("kokoro", settings.kokoro_url, timeout)
    # One conversation turn at a time: the CPU models would only queue anyway,
    # and overlapping turns would interleave conversation history.
    app.state.turn_lock = threading.Lock()
    # MuseTalk avatar (remote GPU service); disabled unless MUSETALK_ENABLED=true.
    app.state.musetalk = avatar.MuseTalkClient(settings)
    app.state.avatar_store = avatar.AvatarStore(settings.output_dir / "avatar_videos", settings.avatar_video_ttl_minutes)
    app.state.perf = PerfLog()
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


def _record_turn(perf: PerfLog, voice_timings: dict, avatar_result: dict | None, total_seconds: float) -> None:
    perf.add("whisper", voice_timings.get("transcription_seconds"))
    perf.add("qwen", voice_timings.get("generation_seconds"))
    perf.add("kokoro", voice_timings.get("synthesis_seconds"))
    if avatar_result and avatar_result.get("status") == "ready":
        perf.add("musetalk", (avatar_result.get("timings") or {}).get("musetalk_render_seconds"))
    perf.add("conversation_turn", total_seconds)


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
    request.app.state.perf.add("whisper", seconds)
    return {
        "transcript": transcript,
        "timings": _round({"transcription_seconds": seconds, "total_seconds": time.perf_counter() - started_at}),
    }


@app.post("/api/qwen/generate")
def qwen_generate(request: Request, body: GenerateRequest):
    """Qwen3 only (AI Lab). With vLLM the reply is streamed internally, which gives
    time to first token and decode speed; the response is still returned complete."""
    started_at = time.perf_counter()
    history = [message.model_dump() for message in body.history] if body.history else None
    result = _call(
        request.app.state.qwen.generate_with_metrics, body.prompt, body.system_prompt, body.max_new_tokens, history
    )
    perf = request.app.state.perf
    perf.add("qwen", result["generation_seconds"])
    perf.add("qwen_ttft", result["time_to_first_token_seconds"])
    timings = {"generation_seconds": result["generation_seconds"], "total_seconds": time.perf_counter() - started_at}
    if result["time_to_first_token_seconds"] is not None:
        timings["time_to_first_token_seconds"] = result["time_to_first_token_seconds"]
    return {
        "response": result["response"],
        "generated_tokens": result["generated_tokens"],
        "backend": result["backend"],
        "finish_reason": result["finish_reason"],
        "tokens_per_second": result["tokens_per_second"],
        "decode_tokens_per_second": result["decode_tokens_per_second"],
        "timings": _round(timings),
    }


@app.post("/api/kokoro/synthesize")
def kokoro_synthesize(request: Request, body: SynthesizeRequest):
    started_at = time.perf_counter()
    wav, duration, seconds = _call(request.app.state.kokoro.synthesize_wav, body.text, body.voice, body.speed)
    request.app.state.perf.add("kokoro", seconds)
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
    _record_turn(state.perf, result.timings, avatar_result, body["timings"]["total_seconds"])
    body.update(
        audio_mime="audio/wav",
        audio_base64=base64.b64encode(result.wav).decode("ascii"),
        audio_duration_seconds=round(result.audio_duration_seconds, 3),
        avatar=avatar_result,
    )
    return body


def _timed(fn):
    started_at = time.perf_counter()
    result = fn()
    return result, round((time.perf_counter() - started_at) * 1000)


def _container_state(health: dict) -> str:
    """What the orchestrator can tell about a container without Docker access.

    It deliberately has no Docker socket (that would give this unauthenticated page
    control of the host), so "running" means "answers on the Docker network".
    """
    status = health.get("status")
    if status == "disabled":
        return "disabled"
    if status == "unreachable":
        return "not_reachable"  # stopped, crashed, or still starting (vLLM: still loading)
    return "running"


@app.get("/api/system/status")
def system_status(request: Request):
    """Everything the System status panel shows, in one call (checks run in parallel)."""
    state = request.app.state
    settings = state.settings
    clients = {"whisper": state.whisper, "qwen": state.qwen, "kokoro": state.kokoro, "musetalk": state.musetalk}
    with ThreadPoolExecutor(max_workers=len(clients) + 3) as pool:
        health_futures = {name: pool.submit(_timed, client.health) for name, client in clients.items()}
        # Whole-GPU numbers from the first GPU container that can run nvidia-smi.
        gpu_futures = [pool.submit(client.gpu) for client in (state.whisper, state.kokoro)]
        engine_future = pool.submit(state.qwen.engine_metrics) if hasattr(state.qwen, "engine_metrics") else None
        services = {}
        for name, future in health_futures.items():
            health, check_ms = future.result()
            services[name] = {**health, "container": _container_state(health), "health_check_ms": check_ms}
        snapshots = [future.result() for future in gpu_futures]
        engine = engine_future.result() if engine_future else None

    gpu = next((snap for snap in snapshots if snap and snap.get("available")), None)
    if gpu is None:
        errors = [snap.get("error") for snap in snapshots if snap and snap.get("error")]
        gpu = {"available": False, "error": errors[0] if errors else "No model service reported GPU metrics."}
    if engine:
        services["qwen"]["engine"] = engine
    if settings.qwen_backend == "vllm" and settings.vllm_gpu_memory_utilization:
        services["qwen"]["gpu_memory_budget_fraction"] = settings.vllm_gpu_memory_utilization
    return {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "qwen_backend": settings.qwen_backend,
        "musetalk_enabled": settings.musetalk_enabled,
        "services": services,
        "gpu": gpu,
        "performance": state.perf.summary(),
    }


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
    if result.get("status") == "ready":
        state.perf.add("musetalk", (result.get("timings") or {}).get("musetalk_render_seconds"))
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
