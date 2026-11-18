"""Kokoro text-to-speech service (English voices; CPU by default, CUDA when device is cuda)."""

import io
import re
import threading
import time
import wave
from pathlib import Path

from . import InvalidInputError, require_module, resolve_device

SAMPLE_RATE = 24000
WEIGHTS_FILE = "kokoro-v1_0.pth"
# The first letter of a Kokoro voice ID is its language code. Only the English
# G2P dependencies (misaki[en] + spaCy en_core_web_sm) are installed.
SUPPORTED_LANG_CODES = {"a": "American English", "b": "British English"}
VOICE_PATTERN = re.compile(r"^[a-z][fm]_[a-z0-9]+$")


def validate_voice(voice: str) -> str:
    voice = voice.strip().lower()
    if not VOICE_PATTERN.match(voice):
        raise InvalidInputError(f"Invalid voice ID '{voice}' (expected a form like af_heart).")
    if voice[0] not in SUPPORTED_LANG_CODES:
        supported = ", ".join(f"{code}* ({name})" for code, name in SUPPORTED_LANG_CODES.items())
        raise InvalidInputError(f"Voice '{voice}' is not an English voice; supported prefixes: {supported}.")
    return voice


def to_wav_bytes(audio) -> bytes:
    """Encode a 1-D float tensor in [-1, 1] as 16-bit PCM mono WAV."""
    torch = require_module("torch")
    pcm = (audio.clamp(-1.0, 1.0) * 32767).round().to(dtype=torch.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.numpy().tobytes())
    return buffer.getvalue()


class KokoroService:
    def __init__(self, repo_id: str, cache_dir: Path, default_voice: str, default_speed: float, device: str = "cpu"):
        self._torch = require_module("torch")
        self._kokoro = require_module("kokoro")
        self._hub = require_module("huggingface_hub")
        self.device = resolve_device(device)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.default_voice = validate_voice(default_voice)
        self.default_speed = default_speed
        self._pipelines = {}  # lang code -> KPipeline (all share one KModel)
        self._voice_paths = {}  # voice ID -> local .pt path
        self._lock = threading.Lock()

        started_at = time.perf_counter()
        config_path = self._download("config.json")
        weights_path = self._download(WEIGHTS_FILE)
        # KPipeline moves each voice pack to model.device and KModel.forward moves its
        # inputs there too, so placing the model on the device puts all inference on it.
        self.model = self._kokoro.KModel(repo_id=repo_id, config=config_path, model=weights_path).to(self.device).eval()
        # Warm the default voice's G2P pipeline and voice pack so the first request is fast.
        self._pipeline(self.default_voice[0])
        self._voice_path(self.default_voice)
        self.load_seconds = time.perf_counter() - started_at

    def _download(self, filename: str) -> str:
        return self._hub.hf_hub_download(self.repo_id, filename, cache_dir=str(self.cache_dir))

    def _pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = self._kokoro.KPipeline(
                lang_code=lang_code, repo_id=self.repo_id, model=self.model, device=self.device
            )
        return self._pipelines[lang_code]

    def _voice_path(self, voice: str) -> str:
        if voice not in self._voice_paths:
            try:
                self._voice_paths[voice] = self._download(f"voices/{voice}.pt")
            except Exception as error:
                raise InvalidInputError(f"Kokoro voice '{voice}' is unavailable: {error}") from error
        return self._voice_paths[voice]

    def synthesize(self, text: str, voice: str | None = None, speed: float | None = None):
        """Return (audio tensor at 24 kHz, seconds). The first use of a new voice downloads ~0.5 MB."""
        text = text.strip()
        if not text:
            raise InvalidInputError("Text must not be empty.")
        voice = validate_voice(voice or self.default_voice)
        speed = self.default_speed if speed is None else speed
        if not 0.5 <= speed <= 2.0:
            raise InvalidInputError("Speed must be between 0.5 and 2.0.")

        with self._lock:
            started_at = time.perf_counter()
            pipeline = self._pipeline(voice[0])
            voice_path = self._voice_path(voice)
            chunks = []
            try:
                with self._torch.inference_mode():
                    for result in pipeline(text, voice=voice_path, speed=speed, split_pattern=r"\n+"):
                        if result.audio is not None:
                            chunks.append(result.audio.detach().cpu().flatten())
            except Exception as error:
                raise RuntimeError(f"Kokoro inference failed: {error}") from error
            seconds = time.perf_counter() - started_at

        if not chunks:
            raise InvalidInputError("Kokoro produced no audio for this text.")
        return self._torch.cat(chunks), seconds

    def synthesize_wav(self, text: str, voice: str | None = None, speed: float | None = None):
        """Return (wav_bytes, audio_duration_seconds, synthesis_seconds)."""
        audio, seconds = self.synthesize(text, voice, speed)
        return to_wav_bytes(audio), len(audio) / SAMPLE_RATE, seconds
