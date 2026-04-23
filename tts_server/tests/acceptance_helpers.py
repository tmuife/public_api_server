from __future__ import annotations

import os
from typing import Any

from fastapi.testclient import TestClient

from main import create_app

DEFAULT_TEST_BASE_URL = "http://testserver/v1"
DEFAULT_TEST_API_KEY = "acceptance-test-token-12345"
DEFAULT_MODEL = "tts-1"
DEFAULT_VOICE = "alloy"


def read_test_base_url() -> str:
    raw_value = os.getenv("TEST_API_BASE_URL", DEFAULT_TEST_BASE_URL).strip()
    if not raw_value:
        return DEFAULT_TEST_BASE_URL
    return raw_value.rstrip("/")


def read_test_api_key() -> str:
    resolved_key = str(os.getenv("TEST_API_KEY", DEFAULT_TEST_API_KEY)).strip()
    if resolved_key:
        return resolved_key
    return DEFAULT_TEST_API_KEY


def configure_test_auth_env() -> str:
    api_key = read_test_api_key()
    os.environ["AUTH_REQUIRED"] = "true"
    os.environ["API_KEY"] = api_key
    return api_key


def build_test_client() -> TestClient:
    configure_test_auth_env()
    return TestClient(create_app(), base_url=read_test_base_url())


def build_auth_headers(api_key: str | None = None) -> dict[str, str]:
    resolved_key = str(api_key or configure_test_auth_env()).strip()
    headers = {"Content-Type": "application/json"}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"
    return headers


def build_speech_payload(
    *,
    model: str = DEFAULT_MODEL,
    input_text: str = "acceptance test request",
    voice: str = DEFAULT_VOICE,
    response_format: str = "wav",
    stream: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": input_text,
        "voice": voice,
        "response_format": response_format,
        "stream": stream,
    }
    payload.update(overrides)
    return payload
