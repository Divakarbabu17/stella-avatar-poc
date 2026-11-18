"""Runtime configuration shared by the API server and the CLIs (read from environment)."""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are Stella, a friendly voice assistant. "
    "Reply in one to three short, natural sentences."
)


QWEN_BACKENDS = ("transformers", "vllm")


def _qwen_backend(value: str) -> str:
    backend = (value or "transformers").strip().lower()
    if backend not in QWEN_BACKENDS:
        raise ValueError(f"QWEN_BACKEND must be one of {', '.join(QWEN_BACKENDS)}, not '{value}'.")
    return backend


@dataclass(frozen=True)
class Settings:
    model_cache_dir: Path
    output_dir: Path
    whisper_model: str
    qwen_model: str
    qwen_system_prompt: str
    qwen_max_new_tokens: int
    kokoro_repo_id: str
    kokoro_voice: str
    kokoro_speed: float
    max_upload_mb: int
    # Torch device per model service: "cpu" (laptop default) or "cuda"/"cuda:N" (compose.gpu.yaml).
    whisper_device: str
    kokoro_device: str
    # Which Qwen service the orchestrator talks to: "transformers" (app.servers.qwen,
    # POST /generate) or "vllm" (OpenAI-compatible /v1/chat/completions).
    qwen_backend: str
    qwen_api_key: str
    # vLLM's --gpu-memory-utilization, only for display in the System status panel (None = unknown).
    vllm_gpu_memory_utilization: float | None
    # Orchestrator -> model service URLs (Docker internal network names).
    whisper_url: str
    qwen_url: str
    kokoro_url: str
    service_timeout_seconds: float
    # Browser conversation: how many earlier messages (user + assistant) reach Qwen3,
    # and the reply-length cap. Both trade context/length for CPU latency.
    conversation_history_messages: int
    conversation_max_new_tokens: int
    # MuseTalk avatar (remote NVIDIA GPU service). Disabled by default: voice-only.
    musetalk_enabled: bool
    musetalk_url: str
    musetalk_timeout_seconds: float
    musetalk_avatar_id: str
    musetalk_api_key: str
    avatar_video_ttl_minutes: float
    avatars_dir: Path

    @property
    def hf_cache_dir(self) -> Path:
        """Hugging Face downloads (Qwen3, Kokoro) live under models/huggingface."""
        return self.model_cache_dir / "huggingface"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            model_cache_dir=Path(env("MODEL_CACHE_DIR", "models")),
            output_dir=Path(env("OUTPUT_DIR", "output")),
            whisper_model=env("WHISPER_MODEL", "base"),
            qwen_model=env("QWEN_MODEL", "Qwen/Qwen3-0.6B"),
            # Compose passes an empty string when the variable is unset; treat that as "use the default".
            qwen_system_prompt=env("QWEN_SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT,
            qwen_max_new_tokens=int(env("QWEN_MAX_NEW_TOKENS", "256")),
            kokoro_repo_id=env("KOKORO_REPO_ID", "hexgrad/Kokoro-82M"),
            kokoro_voice=env("KOKORO_VOICE", "af_heart"),
            kokoro_speed=float(env("KOKORO_SPEED", "1.0")),
            max_upload_mb=int(env("MAX_UPLOAD_MB", "25")),
            whisper_device=env("WHISPER_DEVICE", "cpu").strip().lower() or "cpu",
            kokoro_device=env("KOKORO_DEVICE", "cpu").strip().lower() or "cpu",
            qwen_backend=_qwen_backend(env("QWEN_BACKEND", "transformers")),
            qwen_api_key=env("QWEN_API_KEY", ""),
            vllm_gpu_memory_utilization=float(env("VLLM_GPU_MEMORY_UTILIZATION", "").strip() or 0) or None,
            whisper_url=env("WHISPER_URL", "http://whisper:8001").rstrip("/"),
            qwen_url=env("QWEN_URL", "http://qwen:8002").rstrip("/"),
            kokoro_url=env("KOKORO_URL", "http://kokoro:8003").rstrip("/"),
            service_timeout_seconds=float(env("SERVICE_TIMEOUT_SECONDS", "300")),
            conversation_history_messages=int(env("CONVERSATION_HISTORY_MESSAGES", "6")),
            conversation_max_new_tokens=int(env("CONVERSATION_MAX_NEW_TOKENS", "96")),
            musetalk_enabled=env("MUSETALK_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on"),
            musetalk_url=env("MUSETALK_SERVICE_URL", "http://musetalk:8010").rstrip("/"),
            musetalk_timeout_seconds=float(env("MUSETALK_TIMEOUT", "180")),
            musetalk_avatar_id=env("MUSETALK_AVATAR_ID", "stella"),
            musetalk_api_key=env("MUSETALK_API_KEY", ""),
            avatar_video_ttl_minutes=float(env("AVATAR_VIDEO_TTL_MINUTES", "60")),
            avatars_dir=Path(env("AVATARS_DIR", "avatars")),
        )
