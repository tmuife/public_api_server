from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from app.schemas.openai_compatible import OpenAiModelListResponse, OpenAiSpeechRequest, OpenAiVoiceListResponse
from app.services.openai_speech_service import OpenAiSpeechService


def build_openai_compatible_router(service: OpenAiSpeechService) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["openai-compatible"])

    @router.get("/models", response_model=OpenAiModelListResponse)
    def list_models() -> OpenAiModelListResponse:
        return service.list_models()

    @router.get("/audio/voices", response_model=OpenAiVoiceListResponse)
    def list_voices() -> OpenAiVoiceListResponse:
        return service.list_voices()

    @router.post("/audio/speech")
    async def create_speech(request: Request) -> Response:
        try:
            payload, reference_audio_bytes, reference_audio_file_name = await _parse_speech_request(request)
            result = service.synthesize_audio(
                model=payload.model,
                input_text=payload.input,
                voice=payload.voice,
                response_format=payload.response_format,
                reference_audio_path=payload.reference_audio_path,
                reference_audio_url=payload.reference_audio_url,
                reference_audio_file_bytes=reference_audio_bytes,
                reference_audio_file_name=reference_audio_file_name,
            )
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - runtime safety mapping
            logging.exception("OpenAI compatible speech synthesis failed")
            raise HTTPException(status_code=500, detail="speech synthesis failed") from exc

        if payload.stream:
            return StreamingResponse(
                service.chunk_audio_bytes(result.audio_bytes),
                media_type=result.media_type,
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        return Response(
            content=result.audio_bytes,
            media_type=result.media_type,
            headers={"Cache-Control": "no-cache"},
        )

    return router


async def _parse_speech_request(request: Request) -> tuple[OpenAiSpeechRequest, bytes | None, str | None]:
    content_type = request.headers.get("content-type", "").lower()

    try:
        if content_type.startswith("multipart/form-data") or content_type.startswith(
            "application/x-www-form-urlencoded"
        ):
            return await _parse_form_request(request)

        raw_payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON payload") from exc

    if not isinstance(raw_payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    return _validate_payload(raw_payload), None, None


async def _parse_form_request(request: Request) -> tuple[OpenAiSpeechRequest, bytes | None, str | None]:
    form = await request.form()
    payload_data: dict[str, Any] = {}
    reference_audio_file_bytes: bytes | None = None
    reference_audio_file_name: str | None = None

    for key in (
        "model",
        "input",
        "voice",
        "response_format",
        "stream",
        "reference_audio_url",
        "reference_audio_path",
    ):
        value = form.get(key)
        if value is not None:
            payload_data[key] = value

    upload_file = form.get("reference_audio_file")
    if upload_file is not None:
        if not isinstance(upload_file, UploadFile):
            raise HTTPException(status_code=400, detail="reference_audio_file must be a file upload")
        reference_audio_file_bytes = await upload_file.read()
        reference_audio_file_name = upload_file.filename

    return _validate_payload(payload_data), reference_audio_file_bytes, reference_audio_file_name


def _validate_payload(payload_data: dict[str, Any]) -> OpenAiSpeechRequest:
    try:
        return OpenAiSpeechRequest.model_validate(payload_data)
    except ValidationError as exc:
        first_error = exc.errors()[0]["msg"] if exc.errors() else "invalid request payload"
        raise HTTPException(status_code=400, detail=first_error) from exc
