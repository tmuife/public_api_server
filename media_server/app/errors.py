from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ServiceError(Exception):
    status_code: int
    error_code: str
    message: str
    details: dict[str, Any] | None = None

    def to_http_detail(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload
