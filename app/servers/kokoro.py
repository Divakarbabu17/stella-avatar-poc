"""Kokoro TTS service: `uvicorn app.servers.kokoro:app --port 8003`.

Returns the WAV bytes directly; the orchestrator decides where to store them.
"""

from fastapi.responses import Response

from ..config import Settings
from ..schemas import SynthesizeRequest
from ..services.kokoro import KokoroService
from .common import create_service_app, run_inference

settings = Settings.from_env()
app, holder = create_service_app(
    "kokoro",
    f"{settings.kokoro_repo_id} (default voice {settings.kokoro_voice})",
    lambda: KokoroService(settings.kokoro_repo_id, settings.hf_cache_dir, settings.kokoro_voice, settings.kokoro_speed),
)


@app.post("/synthesize")
def synthesize(body: SynthesizeRequest):
    kokoro = holder.get()
    wav, duration, seconds = run_inference("Synthesis", kokoro.synthesize_wav, body.text, body.voice, body.speed)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "X-Synthesis-Seconds": f"{seconds:.3f}",
            "X-Audio-Duration-Seconds": f"{duration:.3f}",
            "X-Voice": body.voice or kokoro.default_voice,
        },
    )
