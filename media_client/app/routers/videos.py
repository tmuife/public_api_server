from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.errors import ServiceError
from app.schemas import ResponseModel, success_response
from app.services.compose_service import VideoComposeService
from app.services.preprocess_service import VideoPreprocessService

bearer_scheme = HTTPBearer(auto_error=False)


def require_bearer_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    settings = request.app.state.settings
    token = credentials.credentials if credentials else ""
    if not token or token != settings.media_api_access_token:
        raise HTTPException(
            status_code=401,
            detail={"error_code": "unauthorized", "message": "Missing or invalid bearer token"},
        )


router = APIRouter(prefix="/videos", tags=["videos"], dependencies=[Depends(require_bearer_token)])


class ProcessByPathRequest(BaseModel):
    video_path: str = Field(min_length=1)


class ComposeRequest(BaseModel):
    topic: str = Field(min_length=1)


def _preprocess_service(request: Request) -> VideoPreprocessService:
    return request.app.state.preprocess_service


def _compose_service(request: Request) -> VideoComposeService:
    return request.app.state.compose_service


@router.post("/process-by-path", response_model=ResponseModel)
def process_by_path(payload: ProcessByPathRequest, request: Request) -> ResponseModel:
    service = _preprocess_service(request)
    try:
        result = service.process_by_path(payload.video_path)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_http_detail()) from exc

    return success_response(result)


@router.post("/process-upload", response_model=ResponseModel)
async def process_upload(request: Request, file: UploadFile = File(...)) -> ResponseModel:
    service = _preprocess_service(request)
    try:
        result = await service.process_upload(file)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_http_detail()) from exc

    return success_response(result)


@router.post("/compose", response_model=ResponseModel)
def compose_video(payload: ComposeRequest, request: Request) -> ResponseModel:
    service = _compose_service(request)
    try:
        result: dict[str, Any] = service.compose(payload.topic)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_http_detail()) from exc

    return success_response(result)
