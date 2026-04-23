from __future__ import annotations

import secrets
from collections.abc import Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config.auth import AuthConfig

AUTH_CHALLENGE_HEADERS = {"WWW-Authenticate": "Bearer"}
DEFAULT_EXEMPT_PATHS = frozenset({"/health"})
DOCS_EXEMPT_PATHS = frozenset({"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,  # type: ignore[no-untyped-def]
        *,
        auth_config: AuthConfig,
        exempt_paths: Iterable[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._auth_config = auth_config
        self._exempt_paths = frozenset(exempt_paths or DEFAULT_EXEMPT_PATHS)

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        if not self._auth_config.auth_required:
            return await call_next(request)
        if request.url.path in self._exempt_paths:
            return await call_next(request)

        bearer_token = _extract_bearer_token(request.headers.get("Authorization"))
        if bearer_token is None:
            return _unauthorized_response()
        if not secrets.compare_digest(bearer_token, self._auth_config.api_key):
            return _unauthorized_response()

        return await call_next(request)


def _extract_bearer_token(authorization_header: str | None) -> str | None:
    if authorization_header is None:
        return None

    parts = authorization_header.strip().split()
    if len(parts) != 2:
        return None
    if parts[0].lower() != "bearer":
        return None

    token = parts[1].strip()
    return token if token else None


def _unauthorized_response() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"detail": "Unauthorized"},
        headers=AUTH_CHALLENGE_HEADERS,
    )
