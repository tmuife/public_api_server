from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


SUPPORTED_RESPONSE_FORMATS = {"wav", "pcm"}


class OpenAiSpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input: str
    voice: str
    response_format: str = "wav"
    stream: bool = False
    reference_audio_url: str | None = None
    reference_audio_path: str | None = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("model is required")
        return normalized

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("input cannot be empty")
        return normalized

    @field_validator("voice")
    @classmethod
    def validate_voice(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if not normalized:
            raise ValueError("voice is required")
        return normalized

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in SUPPORTED_RESPONSE_FORMATS:
            raise ValueError("response_format must be one of: wav, pcm")
        return normalized

    @field_validator("stream", mode="before")
    @classmethod
    def coerce_stream(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off", ""}:
                return False
        raise ValueError("stream must be a boolean")

    @field_validator("reference_audio_url", "reference_audio_path")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class OpenAiModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class OpenAiModelListResponse(BaseModel):
    object: str = "list"
    data: list[OpenAiModelCard]


class OpenAiVoiceCard(BaseModel):
    canonical: str
    aliases: list[str]
    prompt_audio_file: str | None = None
    prompt_audio_path: str | None = None
    prompt_audio_configured: bool
    prompt_audio_exists: bool


class OpenAiVoiceListResponse(BaseModel):
    object: str = "list"
    data: list[OpenAiVoiceCard]
