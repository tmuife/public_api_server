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
                        "WORK_DIR=/from-env-file",
                        "MEDIA_API_ACCESS_TOKEN=test-token",
                        f"AES_256_GCM_KEY_BASE64={aes_key}",
                        "KAFKA_BOOTSTRAP_SERVERS=broker:9092",
                        "KAFKA_SECURITY_PROTOCOL=PLAINTEXT",
                    ]
                ),
                encoding="utf-8",
            )

            with temporary_env({"WORK_DIR": "/from-system-env"}):
                settings = load_settings(env_file)

            self.assertEqual(str(settings.work_dir), str(Path("/from-system-env").resolve()))
            self.assertEqual(settings.kafka.bootstrap_servers, "broker:9092")

    def test_invalid_aes_key_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"WORK_DIR={temp_dir}",
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

    def test_missing_media_api_access_token_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"WORK_DIR={temp_dir}",
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
                        f"WORK_DIR={temp_dir}",
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

    def test_passphrase_without_salt_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"WORK_DIR={temp_dir}",
                        "MEDIA_API_ACCESS_TOKEN=test-token",
                        "AES_256_GCM_PASSPHRASE=any-length-passphrase",
                        "KAFKA_BOOTSTRAP_SERVERS=broker:9092",
                        "KAFKA_SECURITY_PROTOCOL=PLAINTEXT",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "AES_256_GCM_KDF_SALT_BASE64"):
                load_settings(env_file)

    def test_passphrase_mode_ignores_invalid_base64_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"WORK_DIR={temp_dir}",
                        "MEDIA_API_ACCESS_TOKEN=test-token",
                        "AES_256_GCM_KEY_BASE64=invalid-base64",
                        "AES_256_GCM_PASSPHRASE=my-passphrase",
                        f"AES_256_GCM_KDF_SALT_BASE64={base64.b64encode(b'0123456789abcdef').decode('utf-8')}",
                        "KAFKA_BOOTSTRAP_SERVERS=broker:9092",
                        "KAFKA_SECURITY_PROTOCOL=PLAINTEXT",
                    ]
                ),
                encoding="utf-8",
            )

            settings = load_settings(env_file)

        self.assertEqual(len(settings.aes_key_bytes), 32)


if __name__ == "__main__":
    unittest.main()
