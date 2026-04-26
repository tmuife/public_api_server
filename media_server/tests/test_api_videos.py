from __future__ import annotations

import base64
import os
from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator
import unittest

from fastapi.testclient import TestClient

from app.errors import ServiceError
from main import create_app

ACCESS_TOKEN = "test-bearer-token"


class FakeTopicProcessService:
    def __init__(self):
        self.last_group_id = None

    def dispatch(self, topic: str, group_id: str | None = None):
        if topic == "bad_output":
            raise ServiceError(status_code=400, error_code="invalid_topic", message="topic must end with '_input'")
        self.last_group_id = group_id
        return {
            "job_name": "job_1776945123456_ab12cd34",
            "input_topic": topic,
            "output_topic": "job_1776945123456_ab12cd34_output",
            "group_id": group_id or "media-server-job_1776945123456_ab12cd34",
            "dispatch_mode": "async",
            "status": "accepted",
        }

    def status(self, topic: str, group_id: str | None = None):
        if topic == "missing_input":
            raise ServiceError(status_code=404, error_code="task_not_found", message="No runtime task found for topic")
        return {
            "topic": topic,
            "job_name": "job_1776945123456_ab12cd34",
            "output_topic": "job_1776945123456_ab12cd34_output",
            "group_id": group_id or "media-server-job_1776945123456_ab12cd34",
            "dispatch_mode": "async",
            "status": "completed",
            "consumed_count": 1,
            "published_count": 1,
            "success_count": 1,
            "error_count": 0,
            "error_code": None,
            "error_message": None,
        }


@contextmanager
def temporary_env(values: dict[str, str | None]) -> Iterator[None]:
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


@contextmanager
def api_client() -> Iterator[TestClient]:
    with tempfile.TemporaryDirectory() as temp_dir:
        key_b64 = base64.b64encode(b"a" * 32).decode("utf-8")
        ca_file = Path(temp_dir) / "ca.pem"
        ca_file.write_text("dummy-ca", encoding="utf-8")

        env = {
            "MEDIA_API_ACCESS_TOKEN": ACCESS_TOKEN,
            "AES_256_GCM_KEY_BASE64": key_b64,
            "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            "KAFKA_SECURITY_PROTOCOL": "PLAINTEXT",
        }

        with temporary_env(env):
            app = create_app()
            with TestClient(app) as client:
                client.app.state.topic_process_service = FakeTopicProcessService()
                yield client


class VideoApiTests(unittest.TestCase):
    @staticmethod
    def _auth_headers() -> dict[str, str]:
        return {"Authorization": f"Bearer {ACCESS_TOKEN}"}

    def test_openapi_contains_video_routes(self) -> None:
        with api_client() as client:
            document = client.get("/openapi.json").json()
            self.assertIn("/videos/process-topic", document["paths"])
            self.assertIn("/videos/process-topic/status", document["paths"])
            self.assertIn("HTTPBearer", document["components"]["securitySchemes"])
            security = document["paths"]["/videos/process-topic"]["post"]["security"]
            self.assertEqual(security, [{"HTTPBearer": []}])

    def test_health_is_public(self) -> None:
        with api_client() as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)

    def test_docs_is_public(self) -> None:
        with api_client() as client:
            response = client.get("/docs")
            self.assertEqual(response.status_code, 200)

    def test_protected_route_requires_bearer_token(self) -> None:
        with api_client() as client:
            response = client.post("/videos/process-topic", json={"topic": "job_1776945123456_ab12cd34_input"})
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json()["message"], "unauthorized")

    def test_process_topic_success(self) -> None:
        with api_client() as client:
            response = client.post(
                "/videos/process-topic",
                json={"topic": "job_1776945123456_ab12cd34_input"},
                headers=self._auth_headers(),
            )
            self.assertEqual(response.status_code, 202)
            body = response.json()
            self.assertEqual(body["code"], 0)
            self.assertEqual(body["data"]["status"], "accepted")

    def test_process_topic_accepts_custom_group_id(self) -> None:
        with api_client() as client:
            response = client.post(
                "/videos/process-topic",
                json={"topic": "job_1776945123456_ab12cd34_input", "group_id": "debug-run-001"},
                headers=self._auth_headers(),
            )
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["data"]["group_id"], "debug-run-001")

    def test_process_topic_failure(self) -> None:
        with api_client() as client:
            response = client.post(
                "/videos/process-topic",
                json={"topic": "bad_output"},
                headers=self._auth_headers(),
            )
            self.assertEqual(response.status_code, 400)
            body = response.json()
            self.assertEqual(body["message"], "invalid_topic")

    def test_process_topic_status_success(self) -> None:
        with api_client() as client:
            response = client.get(
                "/videos/process-topic/status",
                params={"topic": "job_1776945123456_ab12cd34_input", "group_id": "debug-run-001"},
                headers=self._auth_headers(),
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["data"]["status"], "completed")
            self.assertEqual(body["data"]["group_id"], "debug-run-001")

    def test_process_topic_status_missing(self) -> None:
        with api_client() as client:
            response = client.get(
                "/videos/process-topic/status",
                params={"topic": "missing_input"},
                headers=self._auth_headers(),
            )
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["message"], "task_not_found")
