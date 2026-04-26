from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn

from app import BearerAuthMiddleware
from app.config import AppSettings, load_settings
from app.routers.videos import router as videos_router
from app.schemas import ResponseModel, success_response
from app.services.frame_processor import PassthroughFrameProcessor
from app.services.kafka_service import KafkaService
from app.services.topic_process_service import TopicProcessService
from app.utils.runtime import RuntimeRegistry

logger = logging.getLogger(__name__)


def _error_response(status_code: int, message: str, detail: Any) -> JSONResponse:
    payload = ResponseModel(code=status_code, message=message, data=detail)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _build_runtime_services(settings: AppSettings) -> tuple[KafkaService, RuntimeRegistry, TopicProcessService]:
    kafka_service = KafkaService(settings.kafka)
    runtime_registry = RuntimeRegistry()
    topic_process_service = TopicProcessService(
        settings=settings,
        kafka_service=kafka_service,
        runtime_registry=runtime_registry,
        frame_processor=PassthroughFrameProcessor(),
    )
    return kafka_service, runtime_registry, topic_process_service


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = load_settings(Path(".env"))
        kafka_service, runtime_registry, topic_process_service = _build_runtime_services(settings)

        app.state.settings = settings
        app.state.kafka_service = kafka_service
        app.state.runtime_registry = runtime_registry
        app.state.topic_process_service = topic_process_service

        logger.info("Runtime settings loaded: %s", json.dumps(settings.redacted(), ensure_ascii=False))
        yield

    app = FastAPI(
        title="media_server API",
        version="0.1.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(BearerAuthMiddleware)

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        message = str(detail.get("error_code") or detail.get("message") or "error")
        return _error_response(status_code=exc.status_code, message=message, detail=detail)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(status_code=400, message="validation_error", detail={"errors": exc.errors()})

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return _error_response(
            status_code=500,
            message="internal_server_error",
            detail={"message": "Unexpected internal error"},
        )

    @app.get("/health", response_model=ResponseModel)
    def health() -> ResponseModel:
        return success_response({"status": "ok"})

    app.include_router(videos_router)
    return app


app = create_app()


def main() -> None:
    host = os.getenv("MEDIA_API_HOST", "0.0.0.0")
    port = int(os.getenv("MEDIA_API_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port)


if __name__ == "__main__":
    main()
