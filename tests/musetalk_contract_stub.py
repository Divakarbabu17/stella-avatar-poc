"""TEST DOUBLE ONLY - NOT MuseTalk and never part of the application.

Implements the HTTP contract of musetalk/service/server.py (/health, /generate,
/videos/{name}) so the CPU laptop can test the orchestrator's success path:
upload of the Kokoro WAV, MP4 download, local storage, browser serving and
deletion on the "GPU" side. /generate returns a plain grey frame muxed with the
received audio. There is no face and no lip-sync; real avatar output requires
the MuseTalk service on an NVIDIA GPU machine.

Run (from the repo root, on the Compose network):
    docker compose --profile cli run -d --name stella-musetalk-stub \
      -v ./tests:/opt/tests:ro -w /opt/tests cli \
      uvicorn musetalk_contract_stub:app --host 0.0.0.0 --port 8010
"""

import os
import shutil
import subprocess
import tempfile
import time
import uuid
import wave
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

API_KEY = os.environ.get("MUSETALK_API_KEY", "stub-key")
OUT = Path(tempfile.gettempdir()) / "stub_videos"
OUT.mkdir(exist_ok=True)
app = FastAPI(title="MuseTalk contract stub (TEST ONLY)")
calls = {"generate": 0, "download": 0, "delete": 0}


def _auth(request: Request) -> None:
    if request.headers.get("authorization") != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


@app.get("/health")
def health(request: Request):
    _auth(request)
    return {"service": "musetalk", "status": "ready", "model": "CONTRACT STUB (not MuseTalk)",
            "model_loaded": True, "cuda_available": False, "gpu_available": False, "calls": calls}


@app.post("/generate")
def generate(request: Request, audio: UploadFile = File(...), avatar_id: str = Form(default="")):
    _auth(request)
    calls["generate"] += 1
    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
        shutil.copyfileobj(audio.file, wav)
        wav.flush()
        with wave.open(wav.name) as w:
            duration = w.getnframes() / w.getframerate()
        name = f"{uuid.uuid4().hex}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=480x640:r=25",
             "-i", wav.name, "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-movflags", "+faststart", str(OUT / name)],
            check=True,
        )
    return {"status": "completed", "avatar_id": avatar_id, "filename": name, "video_url": f"/videos/{name}",
            "audio_duration_seconds": round(duration, 3),
            "timings": {"avatar_prepare_seconds": 0.0, "render_seconds": round(time.perf_counter() - started, 3),
                        "total_seconds": round(time.perf_counter() - started, 3)}}


@app.get("/videos/{name}")
def video(request: Request, name: str):
    _auth(request)
    calls["download"] += 1
    return FileResponse(OUT / name, media_type="video/mp4")


@app.delete("/videos/{name}")
def delete(request: Request, name: str):
    _auth(request)
    calls["delete"] += 1
    (OUT / name).unlink(missing_ok=True)
    return {"deleted": name}
