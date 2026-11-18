"""Model services: each class loads its model once and exposes thread-safe inference.

The orchestrator's HTTP clients expose the same method signatures, so the pipeline
logic in voice.py works with either in-process models or remote services.
"""

import importlib
import re

DEVICE_PATTERN = re.compile(r"^(cpu|cuda(:\d+)?)$")


class InvalidInputError(ValueError):
    """The caller sent unusable input (undecodable audio, empty text, unknown voice)."""


class ServiceUnavailableError(RuntimeError):
    """A model is still loading, failed to load, or its service is not running."""


def require_module(name: str):
    """Import a heavy ML dependency with a clear message when run outside Docker."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        raise RuntimeError(f"Python package '{name}' is unavailable. Run this inside the Docker container.") from error


def resolve_device(requested: str | None) -> str:
    """Validate a torch device setting ("cpu", "cuda" or "cuda:N").

    A CUDA request never falls back to CPU silently: if the container has a
    CPU-only PyTorch build or no GPU access, model loading fails with a clear
    message (surfaced through the service's /health).
    """
    device = (requested or "cpu").strip().lower()
    if not DEVICE_PATTERN.match(device):
        raise ValueError(f"Unsupported device '{requested}' (use cpu, cuda or cuda:N).")
    if device == "cpu":
        return device
    torch = require_module("torch")
    if not torch.cuda.is_available():
        build = f"PyTorch {torch.__version__}, CUDA build {torch.version.cuda or 'none (CPU-only wheel)'}"
        raise RuntimeError(
            f"Device '{device}' requested but CUDA is not available in this container ({build}). "
            "Use the GPU images from compose.gpu.yaml and check the container's GPU access."
        )
    index = int(device.split(":")[1]) if ":" in device else 0
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"Device '{device}' requested but only {torch.cuda.device_count()} CUDA device(s) are visible.")
    return device
