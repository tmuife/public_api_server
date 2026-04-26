from __future__ import annotations

import base64
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest

from app.config import load_settings


@contextmanager
def temporary_env(values: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class SettingsTests(unittest.TestCase):
    def test_env_precedence_over_env_file(self) -> None:
        aes_key = base64.b64encode(b"a" * 32).decode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "MEDIA_API_ACCESS_TOKEN=file-token",
                        f"AES_256_GCM_KEY_BASE64={aes_key}",
                        "KAFKA_BOOTSTRAP_SERVERS=broker:9092",
                        "KAFKA_SECURITY_PROTOCOL=PLAINTEXT",
                    ]
                ),
                encoding="utf-8",
            )

            with temporary_env({"MEDIA_API_ACCESS_TOKEN": "system-token"}):
                settings = load_settings(env_file)

            self.assertEqual(settings.media_api_access_token, "system-token")
            self.assertEqual(settings.kafka.bootstrap_servers, "broker:9092")

    def test_invalid_aes_key_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "MEDIA_API_ACCESS_TOKEN=test-token",
                        "AES_256_GCM_KEY_BASE64=invalid-base64",
                        "KAFKA_BOOTSTRAP_SERVERS=broker:9092",
                        "KAFKA_SECURITY_PROTOCOL=PLAINTEXT",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "AES_256_GCM_KEY_BASE64"):
                load_settings(env_file)

    def test_missing_access_token_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"AES_256_GCM_KEY_BASE64={base64.b64encode(b'a' * 32).decode('utf-8')}",
                        "KAFKA_BOOTSTRAP_SERVERS=broker:9092",
                        "KAFKA_SECURITY_PROTOCOL=PLAINTEXT",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "MEDIA_API_ACCESS_TOKEN"):
                load_settings(env_file)

    def test_passphrase_derives_key(self) -> None:
        passphrase = "lantern-harbor-panda-bamboo-69!"
        salt = b"0123456789abcdef"
        iterations = 100_000
        expected = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, iterations, dklen=32)
        expected_b64 = base64.b64encode(expected).decode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "MEDIA_API_ACCESS_TOKEN=test-token",
                        f"AES_256_GCM_PASSPHRASE={passphrase}",
                        f"AES_256_GCM_KDF_SALT_BASE64={base64.b64encode(salt).decode('utf-8')}",
                        f"AES_256_GCM_KDF_ITERATIONS={iterations}",
                        "KAFKA_BOOTSTRAP_SERVERS=broker:9092",
                        "KAFKA_SECURITY_PROTOCOL=PLAINTEXT",
                    ]
                ),
                encoding="utf-8",
            )

            settings = load_settings(env_file)

        self.assertEqual(settings.aes_key_bytes, expected)
        self.assertEqual(settings.aes_key_b64, expected_b64)

    def test_kafka_consumer_and_producer_auth_are_loaded_separately(self) -> None:
        aes_key = base64.b64encode(b"a" * 32).decode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "MEDIA_API_ACCESS_TOKEN=test-token",
                        f"AES_256_GCM_KEY_BASE64={aes_key}",
                        "KAFKA_BOOTSTRAP_SERVERS=broker:9092",
                        "KAFKA_SECURITY_PROTOCOL=SASL_PLAINTEXT",
                        "KAFKA_SASL_MECHANISM=SCRAM-SHA-512",
                        "KAFKA_SASL_CONSUMER_USERNAME=consumer-user",
                        "KAFKA_SASL_CONSUMER_PASSWORD=consumer-pass",
                        "KAFKA_SASL_PRODUCER_USERNAME=producer-user",
                        "KAFKA_SASL_PRODUCER_PASSWORD=producer-pass",
                    ]
                ),
                encoding="utf-8",
            )

            settings = load_settings(env_file)

        consumer_config = settings.kafka.as_consumer_config()
        producer_config = settings.kafka.as_producer_config()
        admin_config = settings.kafka.as_admin_config()

        self.assertEqual(consumer_config["sasl.username"], "consumer-user")
        self.assertEqual(consumer_config["sasl.password"], "consumer-pass")
        self.assertEqual(producer_config["sasl.username"], "producer-user")
        self.assertEqual(producer_config["sasl.password"], "producer-pass")
        self.assertEqual(admin_config["sasl.username"], "producer-user")
        self.assertEqual(admin_config["sasl.password"], "producer-pass")

    def test_legacy_kafka_sasl_auth_falls_back_for_both_clients(self) -> None:
        aes_key = base64.b64encode(b"a" * 32).decode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "MEDIA_API_ACCESS_TOKEN=test-token",
                        f"AES_256_GCM_KEY_BASE64={aes_key}",
                        "KAFKA_BOOTSTRAP_SERVERS=broker:9092",
                        "KAFKA_SECURITY_PROTOCOL=SASL_PLAINTEXT",
                        "KAFKA_SASL_MECHANISM=SCRAM-SHA-512",
                        "KAFKA_SASL_USERNAME=legacy-user",
                        "KAFKA_SASL_PASSWORD=legacy-pass",
                    ]
                ),
                encoding="utf-8",
            )

            settings = load_settings(env_file)

        self.assertEqual(settings.kafka.as_consumer_config()["sasl.username"], "legacy-user")
        self.assertEqual(settings.kafka.as_consumer_config()["sasl.password"], "legacy-pass")
        self.assertEqual(settings.kafka.as_producer_config()["sasl.username"], "legacy-user")
        self.assertEqual(settings.kafka.as_producer_config()["sasl.password"], "legacy-pass")
