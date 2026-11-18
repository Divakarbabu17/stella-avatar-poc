"""MuseTalk HTTP service (runs on the NVIDIA GPU machine).

    uvicorn service.server:app --host 0.0.0.0 --port 8010   (from /opt/stella)

Endpoints (all require `Authorization: Bearer <MUSETALK_API_KEY>` when the key is set):
    GET    /health                  service, CUDA/GPU and model status
    GET    /avatars                 configured avatars and whether they are prepared
    POST   /avatars/{id}/prepare    prepare an avatar ahead of time
    POST   /generate                WAV (+ avatar_id) -> lip-synced MP4 (metadata + URL)
    GET    /videos/{name}.mp4       download a generated MP4
    DELETE /videos/{name}.mp4       delete it (the orchestrator does this after copying)
"""

import hmac
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .engine import EngineError, InvalidRequestError, MuseTalkEngine, validate_wav

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("musetalk.service")

MUSETALK_DIR = Path(os.environ.get("MUSETALK_DIR", "/opt/MuseTalk"))
AVATARS_DIR = Path(os.environ.get("MUSETALK_AVATARS_DIR", "/data/avatars"))
OUTPUT_DIR = Path(os.environ.get("MUSETALK_OUTPUT_DIR", "/data/outputs"))
DEFAULT_AVATAR = os.environ.get("MUSETALK_DEFAULT_AVATAR", "stella")
API_KEY = os.environ.get("MUSETALK_API_KEY", "")
MAX_AUDIO_MB = float(os.environ.get("MUSETALK_MAX_AUDIO_MB", "20"))
MAX_AUDIO_SECONDS = float(os.environ.get("MUSETALK_MAX_AUDIO_SECONDS", "60"))
OUTPUT_TTL_MINUTES = float(os.environ.get("MUSETALK_OUTPUT_TTL_MINUTES", "30"))
BATCH_SIZE = int(os.environ.get("MUSETALK_BATCH_SIZE", "20"))
GPU_ID = int(os.environ.get("MUSETALK_GPU_ID", "0"))
PREPARE_ON_START = os.environ.get("MUSETALK_PREPARE_ON_START", "true").lower() in ("1", "true", "yes")
VIDEO_NAME = re.compile(r"^[0-9a-f]{32}\.mp4$")

engine = MuseTalkEngine(MUSETALK_DIR, AVATARS_DIR, OUTPUT_DIR, batch_size=BATCH_SIZE, gpu_id=GPU_ID)
status = {"state": "pending", "error": None, "load_seconds": None, "started_at": time.time()}


def _load() -> None:
    status["state"] = "loading"
    started_at = time.perf_counter()
    try:
        engine.load()
        if PREPARE_ON_START:
            status["state"] = "preparing_avatar"
            try:
                engine.prepare(DEFAULT_AVATAR)
            except InvalidRequestError as error:
                # The models work; only the default avatar is missing. Report it but stay usable.
                logger.warning("Default avatar not prepared: %s", error)
                status["error"] = str(error)
    except Exception as error:  # noqa: BLE001 - surfaced through /health
        logger.exception("MuseTalk failed to load")
        status.update(state="error", error=str(error))
        return
    status.update(state="ready", load_seconds=round(time.perf_counter() - started_at, 2))


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not API_KEY:
        logger.warning("MUSETALK_API_KEY is not set: the API is unauthenticated. Only bind it to a private interface.")
    threading.Thread(target=_load, name="musetalk-loader", daemon=True).start()
    yield


app = FastAPI(title="Stella MuseTalk service", version="1.0.0", lifespan=lifespan)


def require_key(request: Request) -> None:
    if not API_KEY:
        return
    supplied = request.headers.get("authorization", "")
    if not hmac.compare_digest(supplied, f"Bearer {API_KEY}"):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


def _cleanup_outputs() -> None:
    cutoff = time.time() - OUTPUT_TTL_MINUTES * 60
    if OUTPUT_DIR.is_dir():
        for path in OUTPUT_DIR.glob("*.mp4"):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)


def _ensure_ready() -> None:
    if status["state"] != "ready":
        detail = status["error"] if status["state"] == "error" else f"MuseTalk is {status['state']}"
        raise HTTPException(status_code=503, detail=detail)


@app.get("/health", dependencies=[Depends(require_key)])
def health():
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        torch_info = {"torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda}
        if cuda_available and engine.loaded:
            # This process's PyTorch GPU memory (orchestrator System status panel).
            torch_info.update(
                gpu_memory_reserved_mb=round(torch.cuda.memory_reserved(GPU_ID) / 2**20, 1),
                gpu_memory_peak_reserved_mb=round(torch.cuda.max_memory_reserved(GPU_ID) / 2**20, 1),
            )
    except Exception as error:  # noqa: BLE001 - torch itself missing/broken
        cuda_available, torch_info = False, {"torch_error": str(error)}
    body = {
        "service": "musetalk",
        "status": status["state"],
        "model": "MuseTalk 1.5 (realtime inference, fp16)",
        "model_loaded": engine.loaded,
        "cuda_available": cuda_available,
        "gpu_available": bool(engine.gpu_info) or cuda_available,
        "gpu": engine.gpu_info or None,
        **torch_info,
        "default_avatar": DEFAULT_AVATAR,
        "load_seconds": status["load_seconds"],
        "uptime_seconds": round(time.time() - status["started_at"], 1),
    }
    if status["error"]:
        body["error"] = status["error"]
    return JSONResponse(body, status_code=200 if status["state"] == "ready" else 503)


@app.get("/avatars", dependencies=[Depends(require_key)])
def avatars():
    return {"default_avatar": DEFAULT_AVATAR, "avatars": engine.list_avatars()}


@app.post("/avatars/{avatar_id}/prepare", dependencies=[Depends(require_key)])
def prepare(avatar_id: str):
    _ensure_ready()
    try:
        seconds = engine.prepare(avatar_id)
    except InvalidRequestError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"avatar_id": avatar_id, "status": "prepared", "seconds": round(seconds, 2)}


@app.post("/generate", dependencies=[Depends(require_key)])
def generate(audio: UploadFile = File(...), avatar_id: str = Form(default="")):
    _ensure_ready()
    _cleanup_outputs()
    avatar_id = (avatar_id or DEFAULT_AVATAR).strip().lower()
    started_at = time.perf_counter()

    # Unique temp file per request; deleted when the request ends, success or not.
    with tempfile.NamedTemporaryFile(suffix=".wav", prefix="musetalk_in_") as temp_file:
        shutil.copyfileobj(audio.file, temp_file, length=1024 * 1024)
        size = temp_file.tell()
        temp_file.flush()
        if size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty.")
        if size > MAX_AUDIO_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Audio exceeds {MAX_AUDIO_MB:.0f} MB.")
        try:
            duration = validate_wav(Path(temp_file.name), MAX_AUDIO_SECONDS)
            result = engine.generate(Path(temp_file.name), avatar_id)
        except InvalidRequestError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except EngineError as error:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(error)) from error

    filename = f"{result['video_id']}.mp4"
    return {
        "status": "completed",
        "avatar_id": avatar_id,
        "filename": filename,
        "video_url": f"/videos/{filename}",
        "audio_duration_seconds": round(duration, 3),
        "video_bytes": result["path"].stat().st_size,
        "timings": {
            "avatar_prepare_seconds": result["avatar_prepare_seconds"],
            "render_seconds": result["render_seconds"],
            "total_seconds": round(time.perf_counter() - started_at, 3),
        },
    }


def _video_path(filename: str) -> Path:
    path = OUTPUT_DIR / filename
    if not VIDEO_NAME.match(filename) or not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found.")
    return path


@app.get("/videos/{filename}", dependencies=[Depends(require_key)])
def get_video(filename: str):
    return FileResponse(_video_path(filename), media_type="video/mp4")


@app.delete("/videos/{filename}", dependencies=[Depends(require_key)])
def delete_video(filename: str):
    _video_path(filename).unlink(missing_ok=True)
    return {"deleted": filename}
