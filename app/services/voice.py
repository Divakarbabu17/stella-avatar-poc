"""Voice pipeline orchestration: audio -> Whisper -> Qwen3 -> (optional) Kokoro.

Works with any objects exposing these methods (in-process services or HTTP clients):
    whisper.transcribe(path) -> (transcript, seconds)
    qwen.generate(prompt, system_prompt, max_new_tokens, history) -> (response, tokens, seconds)
    kokoro.synthesize_wav(text, voice, speed) -> (wav_bytes, duration_seconds, seconds)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import InvalidInputError, ServiceUnavailableError


@dataclass
class VoiceResult:
    transcript: str = ""
    response: str = ""
    wav: bytes | None = None
    audio_duration_seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    # Set when a stage stops the pipeline: "transcription", "generation" or "synthesis".
    failed_stage: str | None = None
    error: str | None = None
    # "invalid_input" (caller's fault), "unavailable" (service not ready) or "failed".
    error_kind: str | None = None
    # True when Whisper decoded the audio but heard no speech.
    no_speech: bool = False


def process_voice(
    audio_path: Path,
    whisper,
    qwen,
    kokoro=None,
    *,
    system_prompt: str | None = None,
    max_new_tokens: int | None = None,
    voice: str | None = None,
    speed: float | None = None,
    history: list[dict] | None = None,
) -> VoiceResult:
    """Run each stage in order and stop at the first failure. Never raises for stage errors."""
    result = VoiceResult()
    started_at = time.perf_counter()

    def stop(stage: str, error: Exception) -> VoiceResult:
        result.failed_stage = stage
        result.error = str(error)
        if isinstance(error, InvalidInputError):
            result.error_kind = "invalid_input"
        elif isinstance(error, ServiceUnavailableError):
            result.error_kind = "unavailable"
        else:
            result.error_kind = "failed"
        result.timings["total_seconds"] = time.perf_counter() - started_at
        return result

    stage_started_at = time.perf_counter()
    try:
        result.transcript, result.timings["transcription_seconds"] = whisper.transcribe(audio_path)
    except Exception as error:
        result.timings["transcription_seconds"] = time.perf_counter() - stage_started_at
        return stop("transcription", error)
    if not result.transcript:
        result.no_speech = True
        return stop("transcription", InvalidInputError("Whisper returned an empty transcript (no speech detected)."))

    try:
        result.response, _, result.timings["generation_seconds"] = qwen.generate(
            result.transcript, system_prompt, max_new_tokens, history
        )
    except Exception as error:
        return stop("generation", error)

    if kokoro is not None:
        try:
            result.wav, result.audio_duration_seconds, result.timings["synthesis_seconds"] = kokoro.synthesize_wav(
                result.response, voice, speed
            )
        except Exception as error:
            return stop("synthesis", error)

    result.timings["total_seconds"] = time.perf_counter() - started_at
    return result
