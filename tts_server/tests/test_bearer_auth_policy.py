from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import create_app

TEST_API_KEY = "policy-test-token-12345"


class BearerAuthPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(
            os.environ,
            {
                "AUTH_REQUIRED": "true",
                "DOCS_PUBLIC_IN_DEV": "false",
                "API_KEY": TEST_API_KEY,
            },
            clear=False,
        )
        self.env_patch.start()
        self.client = TestClient(create_app())
        self.auth_headers = {"Authorization": f"Bearer {TEST_API_KEY}"}

    def tearDown(self) -> None:
        self.client.close()
        self.env_patch.stop()

    def test_health_is_exempt_without_token(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["code"], 0)

    def test_root_docs_redoc_openapi_require_bearer_token(self) -> None:
        for path in ("/", "/docs", "/redoc", "/openapi.json"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.headers.get("www-authenticate"), "Bearer")

    def test_docs_and_openapi_are_accessible_with_valid_token(self) -> None:
        docs_response = self.client.get("/docs", headers=self.auth_headers)
        self.assertEqual(docs_response.status_code, 200)
        self.assertIn("text/html", docs_response.headers.get("content-type", ""))

        openapi_response = self.client.get("/openapi.json", headers=self.auth_headers)
        self.assertEqual(openapi_response.status_code, 200)
        self.assertIn("openapi", openapi_response.json())

    def test_openapi_schema_includes_bearer_security_scheme_for_swagger_authorize(self) -> None:
        openapi_response = self.client.get("/openapi.json", headers=self.auth_headers)
        self.assertEqual(openapi_response.status_code, 200)
        schema = openapi_response.json()

        components = schema.get("components", {})
        security_schemes = components.get("securitySchemes", {})
        bearer_scheme = security_schemes.get("BearerAuth")
        self.assertIsInstance(bearer_scheme, dict)
        self.assertEqual(bearer_scheme.get("type"), "http")
        self.assertEqual(bearer_scheme.get("scheme"), "bearer")

        self.assertEqual(schema.get("security"), [{"BearerAuth": []}])
        self.assertEqual(schema.get("paths", {}).get("/health", {}).get("get", {}).get("security"), [])

    def test_tts_routes_reject_unauthorized_before_business_logic(self) -> None:
        payload = {
            "text": "auth gate check",
            "reference_audio_path": "/tmp/not-used.wav",
        }

        with patch("app.services.tts_service.TtsService.synthesize_batch") as synthesize_batch:
            response = self.client.post("/tts/batch", json=payload)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get("www-authenticate"), "Bearer")
        synthesize_batch.assert_not_called()


class AuthToggleTests(unittest.TestCase):
    def test_auth_required_false_allows_protected_route_without_token(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AUTH_REQUIRED": "false",
                "API_KEY": "API_KEY",
            },
            clear=False,
        ):
            with TestClient(create_app()) as client:
                response = client.get("/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["code"], 0)

    def test_auth_required_true_fails_startup_with_placeholder_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AUTH_REQUIRED": "true",
                "API_KEY": "API_KEY",
            },
            clear=False,
        ):
            with self.assertRaises(RuntimeError):
                with TestClient(create_app()):
                    pass


class DocsPublicInDevToggleTests(unittest.TestCase):
    def test_docs_routes_are_public_when_docs_public_in_dev_enabled(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AUTH_REQUIRED": "true",
                "DOCS_PUBLIC_IN_DEV": "true",
                "API_KEY": TEST_API_KEY,
            },
            clear=False,
        ):
            with TestClient(create_app()) as client:
                for path in ("/docs", "/redoc", "/openapi.json"):
                    with self.subTest(path=path):
                        response = client.get(path)
                        self.assertEqual(response.status_code, 200)

                root_response = client.get("/")
                self.assertEqual(root_response.status_code, 401)
                self.assertEqual(root_response.headers.get("www-authenticate"), "Bearer")

    def test_business_routes_still_require_token_when_docs_public_in_dev_enabled(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AUTH_REQUIRED": "true",
                "DOCS_PUBLIC_IN_DEV": "true",
                "API_KEY": TEST_API_KEY,
            },
            clear=False,
        ):
            with TestClient(create_app()) as client:
                models_response = client.get("/v1/models")
                self.assertEqual(models_response.status_code, 401)
                self.assertEqual(models_response.headers.get("www-authenticate"), "Bearer")


if __name__ == "__main__":
    unittest.main()
