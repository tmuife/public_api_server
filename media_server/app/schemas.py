from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ResponseModel(BaseModel):
    code: int = 0
    message: str = "success"
    data: Any = None


def success_response(data: Any) -> ResponseModel:
    return ResponseModel(code=0, message="success", data=data)
