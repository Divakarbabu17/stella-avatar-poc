"""Shared plumbing for model services: load once at startup, /health, upload and error handling.

Each model service runs as a single uvicorn worker; its model is loaded in a
background thread when the process starts and stays in memory. Route handlers
are sync `def`, so FastAPI runs them in a worker thread and /health stays
responsive while inference is running.
"""

import logging
import os
import shutil
import subprocess
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


def create_service_app(
    name: str, model_label: str, loader: Callable[[], object], device: str | None = None
) -> tuple[FastAPI, ModelHolder]:
    """device (optional): the configured torch device, reported by /health together
    with GPU name and memory once a CUDA model is loaded."""
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
        if device:
            body["device"] = device
            if holder.status == "ready":
                body.update(cuda_status(device))
        if holder.error:
            body["error"] = holder.error
        return JSONResponse(body, status_code=200 if holder.status == "ready" else 503)

    @app.get("/gpu")
    def gpu():
        """Whole-GPU utilization and memory as nvidia-smi reports them (all processes).

        Used by the orchestrator's System status panel. {"available": false} in CPU containers.
        """
        return nvidia_smi_snapshot()

    return app, holder


# nvidia-smi is injected by the NVIDIA Container Toolkit ("utility" capability);
# under WSL2 it lives in /usr/lib/wsl/lib.
NVIDIA_SMI_CANDIDATES = ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi")
GPU_QUERY = (
    ("index", int), ("name", str), ("utilization.gpu", float), ("memory.used", float),
    ("memory.total", float), ("temperature.gpu", float), ("power.draw", float), ("driver_version", str),
)
GPU_KEYS = {
    "utilization.gpu": "utilization_percent", "memory.used": "memory_used_mb", "memory.total": "memory_total_mb",
    "temperature.gpu": "temperature_c", "power.draw": "power_w",
}


def _nvidia_smi_path() -> str | None:
    for candidate in NVIDIA_SMI_CANDIDATES:
        found = shutil.which(candidate) or (candidate if os.path.isfile(candidate) else None)
        if found:
            return found
    return None


def nvidia_smi_snapshot() -> dict:
    executable = _nvidia_smi_path()
    if executable is None:
        return {"available": False, "error": "nvidia-smi not found in this container (CPU mode or no GPU utility access)."}
    fields = ",".join(name for name, _ in GPU_QUERY)
    try:
        result = subprocess.run(
            [executable, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": f"nvidia-smi failed: {error}"}
    if result.returncode != 0:
        return {"available": False, "error": f"nvidia-smi failed: {(result.stderr or result.stdout).strip()[-300:]}"}
    gpus = []
    for line in result.stdout.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(GPU_QUERY):
            continue
        gpu = {}
        for (name, kind), raw in zip(GPU_QUERY, values):
            key = GPU_KEYS.get(name, name)
            try:
                gpu[key] = kind(raw) if raw and not raw.startswith("[") else None  # "[N/A]" -> None
            except ValueError:
                gpu[key] = None
        gpus.append(gpu)
    if not gpus:
        return {"available": False, "error": "nvidia-smi returned no GPUs."}
    return {"available": True, "source": "nvidia-smi", "gpus": gpus}


def cuda_status(device: str) -> dict:
    """GPU name and this process's PyTorch GPU memory (empty for CPU devices).

    reserved = what PyTorch's caching allocator holds (what nvidia-smi attributes
    to the process, minus the CUDA context of roughly 0.3-0.5 GB).
    """
    if not device.startswith("cuda"):
        return {}
    try:
        import torch

        index = int(device.split(":")[1]) if ":" in device else 0
        return {
            "gpu": torch.cuda.get_device_name(index),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(index) / 2**20, 1),
            "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(index) / 2**20, 1),
            "gpu_memory_peak_reserved_mb": round(torch.cuda.max_memory_reserved(index) / 2**20, 1),
        }
    except Exception as error:  # noqa: BLE001 - diagnostic only
        return {"gpu_error": str(error)}


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
