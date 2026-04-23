from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config.env_loader import load_dotenv_into_os_environ


class DotenvLoaderTests(unittest.TestCase):
    def test_loads_supported_keys_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "AUTH_REQUIRED=false\nDOCS_PUBLIC_IN_DEV=true\nAPI_KEY=token-from-dotenv\nAPP_PORT=9000\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                load_dotenv_into_os_environ(env_path)
                self.assertEqual(os.getenv("AUTH_REQUIRED"), "false")
                self.assertEqual(os.getenv("DOCS_PUBLIC_IN_DEV"), "true")
                self.assertEqual(os.getenv("API_KEY"), "token-from-dotenv")
                self.assertEqual(os.getenv("APP_PORT"), "9000")

    def test_does_not_override_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("API_KEY=token-from-dotenv\n", encoding="utf-8")

            with patch.dict(os.environ, {"API_KEY": "token-from-process"}, clear=True):
                load_dotenv_into_os_environ(env_path)
                self.assertEqual(os.getenv("API_KEY"), "token-from-process")

    def test_missing_dotenv_file_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / ".env"
            with patch.dict(os.environ, {}, clear=True):
                load_dotenv_into_os_environ(missing_path)
                self.assertIsNone(os.getenv("API_KEY"))


if __name__ == "__main__":
    unittest.main()
