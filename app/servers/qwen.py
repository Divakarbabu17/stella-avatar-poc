"""Qwen3 service: `uvicorn app.servers.qwen:app --port 8002`."""

from ..config import Settings
from ..schemas import GenerateRequest
from ..services.qwen import QwenService
from .common import create_service_app, run_inference

settings = Settings.from_env()
app, holder = create_service_app(
    "qwen",
    settings.qwen_model,
    lambda: QwenService(
        settings.qwen_model, settings.hf_cache_dir, settings.qwen_system_prompt, settings.qwen_max_new_tokens
    ),
)


@app.post("/generate")
def generate(body: GenerateRequest):
    qwen = holder.get()
    history = [message.model_dump() for message in body.history] if body.history else None
    response, tokens, seconds = run_inference(
        "Generation", qwen.generate, body.prompt, body.system_prompt, body.max_new_tokens, history
    )
    return {
        "response": response,
        "model": qwen.model_name,
        "generated_tokens": tokens,
        "generation_seconds": round(seconds, 3),
    }
