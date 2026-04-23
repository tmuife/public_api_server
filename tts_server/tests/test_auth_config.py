from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.config.auth import AuthConfig, validate_auth_config


class AuthConfigTests(unittest.TestCase):
    def test_parses_auth_required_true_by_default(self) -> None:
        with patch.dict(os.environ, {"API_KEY": "valid-test-token"}, clear=True):
            config = AuthConfig.from_env()

        self.assertTrue(config.auth_required)
        self.assertEqual(config.api_key, "valid-test-token")

    def test_parses_auth_required_false_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AUTH_REQUIRED": "false",
                "API_KEY": "",
            },
            clear=True,
        ):
            config = AuthConfig.from_env()

        self.assertFalse(config.auth_required)
        self.assertEqual(config.api_key, "")
        self.assertFalse(config.docs_public_in_dev)

    def test_parses_docs_public_in_dev_flag(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AUTH_REQUIRED": "true",
                "API_KEY": "valid-test-token",
                "DOCS_PUBLIC_IN_DEV": "true",
            },
            clear=True,
        ):
            config = AuthConfig.from_env()

        self.assertTrue(config.auth_required)
        self.assertTrue(config.docs_public_in_dev)

    def test_validation_rejects_empty_key_when_auth_required(self) -> None:
        config = AuthConfig(api_key="", auth_required=True)
        with self.assertRaises(RuntimeError):
            validate_auth_config(config)

    def test_validation_rejects_placeholder_key_when_auth_required(self) -> None:
        config = AuthConfig(api_key="API_KEY", auth_required=True)
        with self.assertRaises(RuntimeError):
            validate_auth_config(config)

    def test_validation_accepts_empty_key_when_auth_disabled(self) -> None:
        config = AuthConfig(api_key="", auth_required=False)
        validate_auth_config(config)

    def test_validation_accepts_strong_key_when_auth_required(self) -> None:
        config = AuthConfig(api_key="prod-secret-token-123", auth_required=True)
        validate_auth_config(config)


if __name__ == "__main__":
    unittest.main()
