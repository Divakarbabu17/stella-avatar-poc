"""Whisper speech-to-text service (CPU)."""

import threading
import time
from pathlib import Path

from . import InvalidInputError, require_module


class WhisperService:
    def __init__(self, model_name: str, cache_dir: Path):
        self._whisper = require_module("whisper")
        cache_dir.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        self.model = self._whisper.load_model(model_name, device="cpu", download_root=str(cache_dir))
        self.load_seconds = time.perf_counter() - started_at
        self.model_name = model_name
        self._lock = threading.Lock()

    def transcribe(self, audio_path: Path) -> tuple[str, float]:
        """Return (transcript, seconds). Raises InvalidInputError for undecodable/empty audio."""
        started_at = time.perf_counter()
        try:
            # FFmpeg decodes any common format to 16 kHz mono float32.
            audio = self._whisper.load_audio(str(audio_path))
        except Exception as error:
            detail = str(error).strip().splitlines()[-1] if str(error).strip() else type(error).__name__
            raise InvalidInputError(f"Could not decode audio: {detail}") from error
        if audio.size == 0:
            raise InvalidInputError("Audio contains no samples.")

        with self._lock:
            result = self.model.transcribe(audio, fp16=False)
        return result["text"].strip(), time.perf_counter() - started_at
