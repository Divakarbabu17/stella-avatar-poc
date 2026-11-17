"""Whisper service: `uvicorn app.servers.whisper:app --port 8001`."""

from fastapi import File, UploadFile

from ..config import Settings
from ..services.whisper import WhisperService
from .common import create_service_app, run_inference, saved_upload

settings = Settings.from_env()
app, holder = create_service_app(
    "whisper",
    settings.whisper_model,
    lambda: WhisperService(settings.whisper_model, settings.model_cache_dir),
)


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    whisper = holder.get()
    with saved_upload(file, settings.max_upload_mb) as audio_path:
        transcript, seconds = run_inference("Transcription", whisper.transcribe, audio_path)
    return {"transcript": transcript, "model": whisper.model_name, "transcription_seconds": round(seconds, 3)}
