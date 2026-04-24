from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

SENSITIVE_FIELD_MARKERS = (
    "PASSWORD",
    "TOKEN",
    "SECRET",
    "KEY",
)

ALLOWED_SECURITY_PROTOCOLS = {
    "PLAINTEXT",
    "SSL",
    "SASL_PLAINTEXT",
    "SASL_SSL",
}

ALLOWED_SASL_MECHANISMS = {
    "SCRAM-SHA-512",
    "SCRAM-SHA-256",
    "PLAIN",
    "OAUTHBEARER",
    "GSSAPI",
}

DEFAULT_TOPICS = {
    "frame_request": "video.frame.request.v1",
    "frame_result": "video.frame.result.v1",
    "job_event": "video.job.event.v1",
    "dlq": "video.frame.dlq.v1",
    "doctor_rw": "video.doctor.rw.v1",
}


def _normalize_path(value: str, default: str) -> str:
    raw = (value or default).strip()
    if not raw.startswith("/"):
        raw = f"/{raw}"
    return raw


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _read_optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def redact_value(field_name: str, value: Any) -> Any:
    if value is None:
        return None
    upper = field_name.upper()
    if any(marker in upper for marker in SENSITIVE_FIELD_MARKERS):
        return "***"
    return value


def redact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            redacted[key] = redact_mapping(value)
            continue
        if isinstance(value, list):
            redacted[key] = [redact_value(key, item) for item in value]
            continue
        redacted[key] = redact_value(key, value)
    return redacted


@dataclass(frozen=True)
class ApiSettings:
    base_url: str
    health_path: str
    docs_path: str
    openapi_path: str
    docs_oauth2_redirect_path: str
    auth_probe_path: str | None
    timeout_seconds: float
    api_key: str

    @classmethod
    def from_env(cls) -> "ApiSettings":
        base_url = os.getenv("MEDIA_API_BASE_URL", "http://127.0.0.1:8000").strip().rstrip("/")
        if not base_url:
            raise ValueError("MEDIA_API_BASE_URL is required")

        api_key = os.getenv("MEDIA_API_KEY", "").strip()
        if not api_key:
            raise ValueError("MEDIA_API_KEY is required")

        auth_probe_path = _read_optional("MEDIA_API_AUTH_PROBE_PATH")
        if auth_probe_path:
            auth_probe_path = _normalize_path(auth_probe_path, "/")

        return cls(
            base_url=base_url,
            health_path=_normalize_path(os.getenv("MEDIA_API_HEALTH_PATH", ""), "/health"),
            docs_path=_normalize_path(os.getenv("MEDIA_API_DOCS_PATH", ""), "/docs"),
            openapi_path=_normalize_path(os.getenv("MEDIA_API_OPENAPI_PATH", ""), "/openapi.json"),
            docs_oauth2_redirect_path=_normalize_path(
                os.getenv("MEDIA_API_DOCS_OAUTH2_REDIRECT_PATH", ""),
                "/docs/oauth2-redirect",
            ),
            auth_probe_path=auth_probe_path,
            timeout_seconds=_read_float_env("MEDIA_API_TIMEOUT_SECONDS", 10.0),
            api_key=api_key,
        )


@dataclass(frozen=True)
class KafkaSettings:
    bootstrap_servers: str
    security_protocol: str
    sasl_mechanism: str
    sasl_username: str | None
    sasl_password: str | None
    ssl_ca_file: str | None
    topic_frame_request: str
    topic_frame_result: str
    topic_job_event: str
    topic_dlq: str
    topic_doctor_rw: str
    doctor_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "KafkaSettings":
        settings = cls(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").strip(),
            security_protocol=os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL").strip().upper(),
            sasl_mechanism=os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512").strip().upper(),
            sasl_username=_read_optional("KAFKA_SASL_USERNAME"),
            sasl_password=_read_optional("KAFKA_SASL_PASSWORD"),
            ssl_ca_file=_read_optional("KAFKA_SSL_CA_FILE"),
            topic_frame_request=os.getenv("KAFKA_TOPIC_FRAME_REQUEST", DEFAULT_TOPICS["frame_request"]).strip(),
            topic_frame_result=os.getenv("KAFKA_TOPIC_FRAME_RESULT", DEFAULT_TOPICS["frame_result"]).strip(),
            topic_job_event=os.getenv("KAFKA_TOPIC_JOB_EVENT", DEFAULT_TOPICS["job_event"]).strip(),
            topic_dlq=os.getenv("KAFKA_TOPIC_DLQ", DEFAULT_TOPICS["dlq"]).strip(),
            topic_doctor_rw=os.getenv("KAFKA_TOPIC_DOCTOR_RW", DEFAULT_TOPICS["doctor_rw"]).strip(),
            doctor_timeout_seconds=_read_float_env("KAFKA_DOCTOR_TIMEOUT_SECONDS", 15.0),
        )
        settings.validate()
        return settings

    @property
    def topics(self) -> list[str]:
        return [
            self.topic_frame_request,
            self.topic_frame_result,
            self.topic_job_event,
            self.topic_dlq,
            self.topic_doctor_rw,
        ]

    def validate(self) -> None:
        if not self.bootstrap_servers:
            raise ValueError("KAFKA_BOOTSTRAP_SERVERS is required")

        if self.security_protocol not in ALLOWED_SECURITY_PROTOCOLS:
            allowed = ", ".join(sorted(ALLOWED_SECURITY_PROTOCOLS))
            raise ValueError(f"KAFKA_SECURITY_PROTOCOL must be one of: {allowed}")

        if self.sasl_mechanism not in ALLOWED_SASL_MECHANISMS:
            allowed = ", ".join(sorted(ALLOWED_SASL_MECHANISMS))
            raise ValueError(f"KAFKA_SASL_MECHANISM must be one of: {allowed}")

        if self.security_protocol.startswith("SASL"):
            if not self.sasl_username:
                raise ValueError("KAFKA_SASL_USERNAME is required for SASL protocols")
            if not self.sasl_password:
                raise ValueError("KAFKA_SASL_PASSWORD is required for SASL protocols")

        if self.security_protocol in {"SSL", "SASL_SSL"}:
            if not self.ssl_ca_file:
                raise ValueError("KAFKA_SSL_CA_FILE is required when TLS is enabled")

            ca_path = Path(self.ssl_ca_file)
            if not ca_path.exists():
                raise ValueError(f"KAFKA_SSL_CA_FILE not found: {ca_path}")
            if not ca_path.is_file():
                raise ValueError(f"KAFKA_SSL_CA_FILE must be a file: {ca_path}")
            if not os.access(ca_path, os.R_OK):
                raise ValueError(f"KAFKA_SSL_CA_FILE is not readable: {ca_path}")

        for topic_name in self.topics:
            if not topic_name:
                raise ValueError("Kafka topic names must not be empty")


@dataclass(frozen=True)
class AppSettings:
    api: ApiSettings
    kafka: KafkaSettings | None

    @classmethod
    def from_env(cls, require_kafka: bool = True) -> "AppSettings":
        api = ApiSettings.from_env()
        kafka = None
        if require_kafka:
            kafka = KafkaSettings.from_env()
        return cls(api=api, kafka=kafka)

    def redacted(self) -> dict[str, Any]:
        payload = {
            "api": asdict(self.api),
            "kafka": asdict(self.kafka) if self.kafka else None,
        }
        return redact_mapping(payload)
