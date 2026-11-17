"""Request bodies shared by the model services and the orchestrator."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    system_prompt: str | None = Field(default=None, max_length=4000)
    max_new_tokens: int | None = Field(default=None, ge=1, le=2048)
    # Earlier conversation turns, oldest first. Optional; omitted = single-turn.
    history: list[ChatMessage] | None = Field(default=None, max_length=40)


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice: str | None = None
    speed: float | None = Field(default=None, ge=0.5, le=2.0)
