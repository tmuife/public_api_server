from __future__ import annotations

import base64
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

from app.config.env_loader import load_dotenv_into_os_environ
from app.config.auth import AuthConfig, validate_auth_config
from app.middleware.bearer_auth import BearerAuthMiddleware, DEFAULT_EXEMPT_PATHS, DOCS_EXEMPT_PATHS
from app.routers.openai_compatible import build_openai_compatible_router
from app.services.openai_speech_service import build_openai_speech_service
from app.services.tts_service import BatchSynthesisOutput, StreamSynthesisOutput, build_tts_service

load_dotenv_into_os_environ()


class ApiResponse(BaseModel):
    code: int = Field(default=0, description="0 means success")
    message: str = Field(default="success", description="Human-readable message")
    data: Any = Field(default=None, description="Payload")


class HealthData(BaseModel):
    status: str = "ok"
    service: str = "tts-server"


class ServiceInfoData(BaseModel):
    service: str
    version: str
    host: str
    port: int


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, description="Input text to synthesize")
    reference_audio_path: str = Field(min_length=1, description="Local path to reference audio")


class TtsBatchData(BaseModel):
    mode: str
    sample_rate: int
    audio_base64: str
    byte_length: int
    backend: str


class TtsStreamData(BaseModel):
    mode: str
    sample_rate: int
    chunks_base64: list[str]
    chunk_count: int
    aggregated_audio_base64: str
    byte_length: int
    backend: str


def build_success_response(data: Any = None, message: str = "success") -> ApiResponse:
    return ApiResponse(code=0, message=message, data=data)


def _read_host() -> str:
    return os.getenv("APP_HOST", "0.0.0.0")


def _read_port() -> int:
    value = os.getenv("APP_PORT", "8000")
    try:
        return int(value)
    except ValueError:
        logging.warning("Invalid APP_PORT=%s, fallback to 8000", value)
        return 8000


def _encode_base64(raw_bytes: bytes) -> str:
    return base64.b64encode(raw_bytes).decode("ascii")


def _to_batch_data(result: BatchSynthesisOutput) -> TtsBatchData:
    return TtsBatchData(
        mode="batch",
        sample_rate=result.sample_rate,
        audio_base64=_encode_base64(result.audio_bytes),
        byte_length=len(result.audio_bytes),
        backend=result.metadata.backend,
    )


def _to_stream_data(result: StreamSynthesisOutput) -> TtsStreamData:
    return TtsStreamData(
        mode="stream",
        sample_rate=result.sample_rate,
        chunks_base64=[_encode_base64(chunk) for chunk in result.chunk_bytes],
        chunk_count=len(result.chunk_bytes),
        aggregated_audio_base64=_encode_base64(result.aggregated_audio_bytes),
        byte_length=len(result.aggregated_audio_bytes),
        backend=result.metadata.backend,
    )


def _build_auth_exempt_paths(auth_config: AuthConfig) -> frozenset[str]:
    exempt_paths = set(DEFAULT_EXEMPT_PATHS)
    if auth_config.docs_public_in_dev:
        exempt_paths.update(DOCS_EXEMPT_PATHS)
    return frozenset(exempt_paths)


def _configure_openapi_security(app: FastAPI) -> None:
    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        components = schema.setdefault("components", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API Key",
        }
        schema["security"] = [{"BearerAuth": []}]

        # Keep docs aligned with runtime policy: /health is the only auth-exempt endpoint.
        health_get = schema.get("paths", {}).get("/health", {}).get("get")
        if isinstance(health_get, dict):
            health_get["security"] = []

        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


def create_app(auth_config: AuthConfig | None = None) -> FastAPI:
    resolved_auth_config = auth_config or AuthConfig.from_env()
    app = FastAPI(title="TTS Server", version="0.2.0")
    app.add_middleware(
        BearerAuthMiddleware,
        auth_config=resolved_auth_config,
        exempt_paths=_build_auth_exempt_paths(resolved_auth_config),
    )

    tts_service = build_tts_service()
    openai_service = build_openai_speech_service(tts_service)

    @app.on_event("startup")
    def startup_event() -> None:
        validate_auth_config(resolved_auth_config)
        tts_service.initialize()

    @app.get("/", response_model=ApiResponse)
    def get_service_info() -> ApiResponse:
        return build_success_response(
            ServiceInfoData(
                service="tts-server",
                version=app.version,
                host=_read_host(),
                port=_read_port(),
            )
        )

    @app.get("/health", response_model=ApiResponse)
    def get_health() -> ApiResponse:
        return build_success_response(HealthData())

    @app.post("/tts/batch", response_model=ApiResponse)
    def synthesize_batch(payload: TtsRequest) -> ApiResponse:
        try:
            result = tts_service.synthesize_batch(
                text=payload.text,
                reference_audio_path=payload.reference_audio_path,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return build_success_response(_to_batch_data(result))

    @app.post("/tts/stream", response_model=ApiResponse)
    def synthesize_stream(payload: TtsRequest) -> ApiResponse:
        try:
            result = tts_service.synthesize_stream(
                text=payload.text,
                reference_audio_path=payload.reference_audio_path,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return build_success_response(_to_stream_data(result))

    app.include_router(build_openai_compatible_router(openai_service))
    _configure_openapi_security(app)

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import uvicorn

    uvicorn.run("main:app", host=_read_host(), port=_read_port(), reload=False)
