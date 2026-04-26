from __future__ import annotations

from typing import Any, Mapping

FRAME_REQUEST_REQUIRED_FIELDS = {
    "job_name",
    "frame_index",
    "nonce_b64",
    "ciphertext_b64",
    "tag_b64",
    "content_type",
}

FRAME_RESULT_REQUIRED_FIELDS = {
    "job_name",
    "frame_index",
    "status",
    "nonce_b64",
    "ciphertext_b64",
    "tag_b64",
}

FORBIDDEN_SECRET_FIELD_MARKERS = (
    "token",
    "password",
    "secret",
    "master_key",
    "wrapped_key",
    "key",
    "credential",
)


def _validate_secret_markers(payload: Mapping[str, Any], *, source: str) -> None:
    for key in payload.keys():
        lowered = key.lower()
        if any(marker in lowered for marker in FORBIDDEN_SECRET_FIELD_MARKERS):
            raise ValueError(f"Forbidden secret field in {source}: {key}")


def _validate_frame_index(payload: Mapping[str, Any]) -> None:
    frame_index = payload.get("frame_index")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise ValueError("frame_index must be an integer")
    if frame_index < 0:
        raise ValueError("frame_index must be zero-based (>= 0)")


def validate_kafka_publication(payload: Mapping[str, Any], *, headers: Mapping[str, Any] | None = None) -> None:
    _validate_secret_markers(payload, source="Kafka payload")
    if headers:
        _validate_secret_markers(headers, source="Kafka headers")


def validate_frame_request(payload: Mapping[str, Any]) -> None:
    validate_kafka_publication(payload)
    missing = sorted(FRAME_REQUEST_REQUIRED_FIELDS.difference(payload.keys()))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Invalid frame request payload, missing fields: {joined}")
    _validate_frame_index(payload)


def validate_frame_result(payload: Mapping[str, Any]) -> None:
    validate_kafka_publication(payload)
    missing = sorted(FRAME_RESULT_REQUIRED_FIELDS.difference(payload.keys()))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Invalid frame result payload, missing fields: {joined}")
    _validate_frame_index(payload)

    status = payload.get("status")
    if not isinstance(status, str) or status not in {"ok", "error"}:
        raise ValueError("status must be either 'ok' or 'error'")

    if status == "error":
        error_code = payload.get("error_code")
        if not isinstance(error_code, str) or not error_code.strip():
            raise ValueError("error payload must include non-empty error_code")
