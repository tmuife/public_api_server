from __future__ import annotations

from typing import Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.schemas import ResponseModel


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, excluded_paths: Iterable[str] | None = None):
        super().__init__(app)
        self.excluded_paths = tuple(excluded_paths or ())

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if self._is_excluded(path):
            return await call_next(request)

        settings = getattr(request.app.state, "settings", None)
        expected_token = getattr(settings, "media_api_access_token", None)
        if not expected_token:
            payload = ResponseModel(
                code=500,
                message="internal_server_error",
                data={"message": "MEDIA_API_ACCESS_TOKEN is not configured"},
            )
            return Response(
                content=payload.model_dump_json(),
                status_code=500,
                media_type="application/json",
            )

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return self._unauthorized_response()

        token = auth_header[len("Bearer ") :].strip()
        if token != expected_token:
            return self._unauthorized_response()

        return await call_next(request)

    def _is_excluded(self, path: str) -> bool:
        if path in self.excluded_paths:
            return True
        if path in {"/health", "/openapi.json"}:
            return True
        if path == "/docs" or path.startswith("/docs/"):
            return True
        return False

    @staticmethod
    def _unauthorized_response() -> Response:
        payload = ResponseModel(
            code=401,
            message="unauthorized",
            data={
                "error_code": "unauthorized",
                "message": "Missing or invalid bearer token",
            },
        )
        return Response(
            content=payload.model_dump_json(),
            status_code=401,
            media_type="application/json",
            headers={"WWW-Authenticate": "Bearer"},
        )
