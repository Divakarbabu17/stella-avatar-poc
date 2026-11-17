"""Model services: each class loads its model once and exposes thread-safe inference.

The orchestrator's HTTP clients expose the same method signatures, so the pipeline
logic in voice.py works with either in-process models or remote services.
"""

import importlib


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
