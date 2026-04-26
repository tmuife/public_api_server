from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from decouple import Config, RepositoryEnv, UndefinedValueError

T = TypeVar("T")

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

SENSITIVE_MARKERS = ("PASSWORD", "TOKEN", "SECRET", "KEY")


class _SettingsLoader:
    def __init__(self, env_path: Path | None = None):
        self.env_path = env_path or Path(".env")
        self.file_config = self._build_file_config(self.env_path)

    @staticmethod
    def _build_file_config(env_path: Path) -> Config | None:
        if not env_path.exists():
            return None
        return Config(RepositoryEnv(str(env_path)))

    def read(
        self,
        name: str,
        *,
        default: T | None = None,
        cast: Callable[[str], T] | None = None,
        required: bool = False,
    ) -> T | None:
        raw = os.getenv(name)

        if raw is None and self.file_config is not None:
            try:
                raw = self.file_config(name)
            except UndefinedValueError:
                raw = None

        if raw is None:
            if required:
                raise ValueError(f"{name} is required")
            return default

        value = raw.strip()
        if required and not value:
            raise ValueError(f"{name} is required")

        if cast is None:
            return value  # type: ignore[return-value]

        try:
            return cast(value)
        except Exception as exc:
            raise ValueError(f"{name} has invalid value") from exc


def _parse_positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError("value must be > 0")
    return parsed


def _parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("value must be > 0")
    return parsed


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _decode_aes_key(aes_key_b64: str) -> bytes:
    try:
        decoded = base64.b64decode(aes_key_b64, validate=True)
    except Exception as exc:
        raise ValueError("AES_256_GCM_KEY_BASE64 must be valid base64") from exc

    if len(decoded) != 32:
        raise ValueError("AES_256_GCM_KEY_BASE64 must decode to 32 bytes")
    return decoded


def _derive_aes_key_from_passphrase(
    *,
    passphrase: str,
    salt_b64: str,
    iterations: int,
) -> tuple[str, bytes]:
    if iterations < 100_000:
        raise ValueError("AES_256_GCM_KDF_ITERATIONS must be >= 100000")

    try:
        salt = base64.b64decode(salt_b64, validate=True)
    except Exception as exc:
        raise ValueError("AES_256_GCM_KDF_SALT_BASE64 must be valid base64") from exc

    if len(salt) < 16:
        raise ValueError("AES_256_GCM_KDF_SALT_BASE64 must decode to at least 16 bytes")

    key_bytes = hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        iterations,
        dklen=32,
    )
    return base64.b64encode(key_bytes).decode("utf-8"), key_bytes


def _redact_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    upper = field.upper()
    if any(marker in upper for marker in SENSITIVE_MARKERS):
        return "***"
    return value


def redact_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            redacted[key] = redact_mapping(value)
            continue
        if isinstance(value, list):
            redacted[key] = [_redact_value(key, item) for item in value]
            continue
        redacted[key] = _redact_value(key, value)
    return redacted


@dataclass(frozen=True)
class KafkaAuthSettings:
    sasl_username: str | None = None
    sasl_password: str | None = None


@dataclass(frozen=True)
class KafkaSettings:
    bootstrap_servers: str
    security_protocol: str
    sasl_mechanism: str | None
    ssl_ca_file: str | None
    produce_timeout_seconds: float
    wait_first_frame_seconds: float
    idle_timeout_seconds: float
    max_process_seconds: float
    topic_partitions: int
    topic_replication_factor: int
    consumer_auth: KafkaAuthSettings = field(default_factory=KafkaAuthSettings)
    producer_auth: KafkaAuthSettings = field(default_factory=KafkaAuthSettings)

    def validate(self) -> None:
        if not self.bootstrap_servers:
            raise ValueError("KAFKA_BOOTSTRAP_SERVERS is required")

        if self.security_protocol not in ALLOWED_SECURITY_PROTOCOLS:
            allowed = ", ".join(sorted(ALLOWED_SECURITY_PROTOCOLS))
            raise ValueError(f"KAFKA_SECURITY_PROTOCOL must be one of: {allowed}")

        if self.security_protocol.startswith("SASL"):
            if not self.sasl_mechanism:
                raise ValueError("KAFKA_SASL_MECHANISM is required for SASL protocols")
            if self.sasl_mechanism not in ALLOWED_SASL_MECHANISMS:
                allowed = ", ".join(sorted(ALLOWED_SASL_MECHANISMS))
                raise ValueError(f"KAFKA_SASL_MECHANISM must be one of: {allowed}")
            self._validate_auth(self.consumer_auth, prefix="KAFKA_SASL_CONSUMER")
            self._validate_auth(self.producer_auth, prefix="KAFKA_SASL_PRODUCER")

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

    @staticmethod
    def _validate_auth(auth: KafkaAuthSettings, *, prefix: str) -> None:
        if not auth.sasl_username:
            raise ValueError(f"{prefix}_SASL_USERNAME is required for SASL protocols")
        if not auth.sasl_password:
            raise ValueError(f"{prefix}_SASL_PASSWORD is required for SASL protocols")

    def _common_client_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "bootstrap.servers": self.bootstrap_servers,
            "security.protocol": self.security_protocol,
        }
        if self.sasl_mechanism:
            config["sasl.mechanism"] = self.sasl_mechanism
        if self.security_protocol in {"SSL", "SASL_SSL"}:
            config["ssl.ca.location"] = self.ssl_ca_file
        return config

    def _client_config_with_auth(self, auth: KafkaAuthSettings) -> dict[str, Any]:
        config = self._common_client_config()
        if self.security_protocol.startswith("SASL"):
            config["sasl.username"] = auth.sasl_username
            config["sasl.password"] = auth.sasl_password
        return config

    def as_consumer_config(self) -> dict[str, Any]:
        return self._client_config_with_auth(self.consumer_auth)

    def as_producer_config(self) -> dict[str, Any]:
        return self._client_config_with_auth(self.producer_auth)

    def as_admin_config(self) -> dict[str, Any]:
        return self.as_producer_config()

    def as_client_config(self) -> dict[str, Any]:
        return self.as_producer_config()


@dataclass(frozen=True)
class AppSettings:
    media_api_access_token: str
    aes_key_b64: str
    aes_key_bytes: bytes
    kafka: KafkaSettings

    def redacted(self) -> dict[str, Any]:
        return redact_mapping(
            {
                "media_api_access_token": self.media_api_access_token,
                "aes_key_b64": self.aes_key_b64,
                "kafka": asdict(self.kafka),
            }
        )


def load_settings(env_path: Path | None = None) -> AppSettings:
    loader = _SettingsLoader(env_path)

    media_api_access_token = loader.read("MEDIA_API_ACCESS_TOKEN", required=True)
    if media_api_access_token is None:
        raise ValueError("MEDIA_API_ACCESS_TOKEN is required")

    aes_passphrase = _normalize_optional(loader.read("AES_256_GCM_PASSPHRASE", default=""))
    if aes_passphrase:
        salt_b64 = loader.read("AES_256_GCM_KDF_SALT_BASE64", required=True)
        if salt_b64 is None:
            raise ValueError("AES_256_GCM_KDF_SALT_BASE64 is required")
        iterations = loader.read("AES_256_GCM_KDF_ITERATIONS", default=600_000, cast=_parse_positive_int) or 600_000
        aes_key_b64, aes_key_bytes = _derive_aes_key_from_passphrase(
            passphrase=aes_passphrase,
            salt_b64=salt_b64,
            iterations=iterations,
        )
    else:
        aes_key_b64 = _normalize_optional(loader.read("AES_256_GCM_KEY_BASE64", default=""))
        if not aes_key_b64:
            raise ValueError("Either AES_256_GCM_KEY_BASE64 or AES_256_GCM_PASSPHRASE must be provided")
        aes_key_bytes = _decode_aes_key(aes_key_b64)

    legacy_sasl_username = _normalize_optional(loader.read("KAFKA_SASL_USERNAME", default=""))
    legacy_sasl_password = _normalize_optional(loader.read("KAFKA_SASL_PASSWORD", default=""))
    consumer_auth = KafkaAuthSettings(
        sasl_username=_normalize_optional(loader.read("KAFKA_SASL_CONSUMER_USERNAME", default=""))
        or _normalize_optional(loader.read("KAFKA_CONSUMER_SASL_USERNAME", default=""))
        or legacy_sasl_username,
        sasl_password=_normalize_optional(loader.read("KAFKA_SASL_CONSUMER_PASSWORD", default=""))
        or _normalize_optional(loader.read("KAFKA_CONSUMER_SASL_PASSWORD", default=""))
        or legacy_sasl_password,
    )
    producer_auth = KafkaAuthSettings(
        sasl_username=_normalize_optional(loader.read("KAFKA_SASL_PRODUCER_USERNAME", default=""))
        or _normalize_optional(loader.read("KAFKA_PRODUCER_SASL_USERNAME", default=""))
        or legacy_sasl_username,
        sasl_password=_normalize_optional(loader.read("KAFKA_SASL_PRODUCER_PASSWORD", default=""))
        or _normalize_optional(loader.read("KAFKA_PRODUCER_SASL_PASSWORD", default=""))
        or legacy_sasl_password,
    )

    kafka = KafkaSettings(
        bootstrap_servers=loader.read("KAFKA_BOOTSTRAP_SERVERS", required=True) or "",
        security_protocol=(loader.read("KAFKA_SECURITY_PROTOCOL", default="SASL_SSL") or "SASL_SSL").upper(),
        sasl_mechanism=_normalize_optional(
            (loader.read("KAFKA_SASL_MECHANISM", default="SCRAM-SHA-512") or "SCRAM-SHA-512").upper()
        ),
        ssl_ca_file=_normalize_optional(loader.read("KAFKA_SSL_CA_FILE", default="")),
        produce_timeout_seconds=loader.read(
            "KAFKA_PRODUCE_TIMEOUT_SECONDS",
            default=30.0,
            cast=_parse_positive_float,
        )
        or 30.0,
        wait_first_frame_seconds=loader.read(
            "KAFKA_WAIT_FIRST_FRAME_SECONDS",
            default=10.0,
            cast=_parse_positive_float,
        )
        or 10.0,
        idle_timeout_seconds=loader.read(
            "KAFKA_IDLE_TIMEOUT_SECONDS",
            default=3.0,
            cast=_parse_positive_float,
        )
        or 3.0,
        max_process_seconds=loader.read(
            "KAFKA_MAX_PROCESS_SECONDS",
            default=120.0,
            cast=_parse_positive_float,
        )
        or 120.0,
        topic_partitions=loader.read("KAFKA_TOPIC_PARTITIONS", default=1, cast=_parse_positive_int) or 1,
        topic_replication_factor=loader.read(
            "KAFKA_TOPIC_REPLICATION_FACTOR",
            default=1,
            cast=_parse_positive_int,
        )
        or 1,
        consumer_auth=consumer_auth,
        producer_auth=producer_auth,
    )
    kafka.validate()

    return AppSettings(
        media_api_access_token=media_api_access_token,
        aes_key_b64=aes_key_b64,
        aes_key_bytes=aes_key_bytes,
        kafka=kafka,
    )
