"""Shared plumbing for model services: load once at startup, /health, upload and error handling.

Each model service runs as a single uvicorn worker; its model is loaded in a
background thread when the process starts and stays in memory. Route handlers
are sync `def`, so FastAPI runs them in a worker thread and /health stays
responsive while inference is running.
"""

import logging
import os
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..services import InvalidInputError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class ModelHolder:
    def __init__(self, name: str, model_label: str, loader: Callable[[], object]):
        self.name = name
        self.model_label = model_label
        self._loader = loader
        self._service = None
        self._lock = threading.Lock()
        self.status = "pending"
        self.error: str | None = None
        self.load_seconds: float | None = None
        self.loaded_at: str | None = None
        self.load_count = 0
        self.logger = logging.getLogger(f"stella.{name}")

    def load(self) -> None:
        self.status = "loading"
        self.logger.info("Loading %s model %s", self.name, self.model_label)
        started_at = time.perf_counter()
        try:
            service = self._loader()
        except Exception as error:
            self.logger.exception("Failed to load %s model", self.name)
            self.status, self.error = "error", str(error)
            return
        with self._lock:
            self._service = service
            self.load_count += 1
            self.load_seconds = round(time.perf_counter() - started_at, 2)
            self.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.status = "ready"
        self.logger.info("Loaded %s in %.2f seconds", self.name, self.load_seconds)

    def get(self):
        """Return the loaded service or raise HTTP 503 while loading / after a load failure."""
        if self._service is None:
            detail = f"{self.name} model failed to load: {self.error}" if self.error else f"{self.name} model is {self.status}"
            raise HTTPException(status_code=503, detail=detail)
        return self._service


def create_service_app(name: str, model_label: str, loader: Callable[[], object]) -> tuple[FastAPI, ModelHolder]:
    holder = ModelHolder(name, model_label, loader)
    started_at = time.time()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        threading.Thread(target=holder.load, name=f"{name}-loader", daemon=True).start()
        yield

    app = FastAPI(title=f"Stella {name} service", lifespan=lifespan)

    @app.get("/health")
    def health():
        body = {
            "service": name,
            "status": holder.status,
            "model": model_label,
            "load_seconds": holder.load_seconds,
            "loaded_at": holder.loaded_at,
            "load_count": holder.load_count,
            "pid": os.getpid(),
            "uptime_seconds": round(time.time() - started_at, 1),
            "memory_rss_mb": rss_mb(),
        }
        if holder.error:
            body["error"] = holder.error
        return JSONResponse(body, status_code=200 if holder.status == "ready" else 503)

    return app, holder


def rss_mb() -> float | None:
    """Resident memory of this process (Linux /proc), in MiB."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


@contextmanager
def saved_upload(upload: UploadFile, max_upload_mb: int):
    """Copy an upload to a temp file (FFmpeg needs a path), enforcing the size limit."""
    limit = max_upload_mb * 1024 * 1024
    suffix = Path(upload.filename or "").suffix[:10]
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
        shutil.copyfileobj(upload.file, temp_file, length=1024 * 1024)
        size = temp_file.tell()
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        if size > limit:
            raise HTTPException(status_code=413, detail=f"Audio file exceeds {max_upload_mb} MB.")
        temp_file.flush()
        yield Path(temp_file.name)


def run_inference(label: str, fn: Callable, *args):
    """Map service exceptions to HTTP errors: bad input -> 400, anything else -> 500."""
    try:
        return fn(*args)
    except InvalidInputError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logging.getLogger("stella").exception("%s failed", label)
        raise HTTPException(status_code=500, detail=f"{label} failed: {error}") from error
