from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.errors import ServiceError
from app.schemas import ResponseModel, success_response
from app.services.topic_process_service import TopicProcessService

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


class ProcessTopicRequest(BaseModel):
    topic: str = Field(min_length=1)
    group_id: str | None = Field(default=None, min_length=1)


def _topic_process_service(request: Request) -> TopicProcessService:
    return request.app.state.topic_process_service


@router.post("/process-topic", response_model=ResponseModel, status_code=202)
def process_topic(payload: ProcessTopicRequest, request: Request) -> ResponseModel:
    service = _topic_process_service(request)
    try:
        result = service.dispatch(payload.topic, payload.group_id)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_http_detail()) from exc

    return success_response(result)


@router.get("/process-topic/status", response_model=ResponseModel)
def process_topic_status(
    request: Request,
    topic: str = Query(min_length=1),
    group_id: str | None = Query(default=None, min_length=1),
) -> ResponseModel:
    service = _topic_process_service(request)
    try:
        result = service.status(topic, group_id)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_http_detail()) from exc

    return success_response(result)
