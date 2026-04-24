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


class FakePreprocessService:
    def process_by_path(self, video_path: str):
        if video_path == "/not/exist.mp4":
            raise ServiceError(
                status_code=400,
                error_code="invalid_video_path",
                message="Video path does not exist or is not a file",
            )
        return {
            "job_name": "job_1776945123456_ab12cd34",
            "input_topic": "job_1776945123456_ab12cd34_input",
            "manifest_summary": {
                "frame_count": 2,
                "fps": 24.0,
                "bitrate": 1000000,
                "frame_format": "jpg",
                "audio_path": "/tmp/audio.m4a",
            },
        }

    async def process_upload(self, upload_file):
        if not upload_file.filename:
            raise ServiceError(status_code=400, error_code="invalid_upload", message="Upload filename is required")
        return {
            "job_name": "job_1776945123456_ab12cd34",
            "input_topic": "job_1776945123456_ab12cd34_input",
            "manifest_summary": {
                "frame_count": 2,
                "fps": 24.0,
                "bitrate": 1000000,
                "frame_format": "jpg",
                "audio_path": "/tmp/audio.m4a",
            },
        }


class FakeComposeService:
    def compose(self, topic: str):
        if topic == "job_missing_output":
            raise ServiceError(status_code=404, error_code="job_not_found", message="No workspace found for topic")
        return {
            "job_name": "job_1776945123456_ab12cd34",
            "topic": topic,
            "output_path": "/tmp/workdir/jobs/job_1776945123456_ab12cd34/output/final.mp4",
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
        work_dir = Path(temp_dir) / "workdir"
        upload_dir = work_dir / "uploads"
        key_b64 = base64.b64encode(b"a" * 32).decode("utf-8")

        env = {
            "MEDIA_API_ACCESS_TOKEN": ACCESS_TOKEN,
            "WORK_DIR": str(work_dir),
            "UPLOAD_DIR": str(upload_dir),
            "AES_256_GCM_KEY_BASE64": key_b64,
            "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            "KAFKA_SECURITY_PROTOCOL": "PLAINTEXT",
        }

        with temporary_env(env):
            app = create_app()
            with TestClient(app) as client:
                client.app.state.preprocess_service = FakePreprocessService()
                client.app.state.compose_service = FakeComposeService()
                yield client


class VideoApiTests(unittest.TestCase):
    @staticmethod
    def _auth_headers() -> dict[str, str]:
        return {"Authorization": f"Bearer {ACCESS_TOKEN}"}

    def test_openapi_contains_video_routes(self) -> None:
        with api_client() as client:
            document = client.get("/openapi.json").json()
            self.assertIn("/videos/process-by-path", document["paths"])
            self.assertIn("/videos/process-upload", document["paths"])
            self.assertIn("/videos/compose", document["paths"])
            self.assertIn("HTTPBearer", document["components"]["securitySchemes"])
            security = document["paths"]["/videos/process-by-path"]["post"]["security"]
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
            response = client.post("/videos/process-by-path", json={"video_path": "/tmp/source.mp4"})
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json()["message"], "unauthorized")

    def test_process_by_path_success(self) -> None:
        with api_client() as client:
            response = client.post(
                "/videos/process-by-path",
                json={"video_path": "/tmp/source.mp4"},
                headers=self._auth_headers(),
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["code"], 0)
            self.assertEqual(body["data"]["job_name"], "job_1776945123456_ab12cd34")

    def test_process_by_path_failure(self) -> None:
        with api_client() as client:
            response = client.post(
                "/videos/process-by-path",
                json={"video_path": "/not/exist.mp4"},
                headers=self._auth_headers(),
            )
            self.assertEqual(response.status_code, 400)
            body = response.json()
            self.assertEqual(body["message"], "invalid_video_path")

    def test_process_upload_success(self) -> None:
        with api_client() as client:
            response = client.post(
                "/videos/process-upload",
                files={"file": ("demo.mp4", b"video-bytes", "video/mp4")},
                headers=self._auth_headers(),
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["code"], 0)

    def test_compose_success(self) -> None:
        with api_client() as client:
            response = client.post(
                "/videos/compose",
                json={"topic": "job_1776945123456_ab12cd34_output"},
                headers=self._auth_headers(),
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["data"]["output_path"].endswith("final.mp4"))

    def test_compose_failure(self) -> None:
        with api_client() as client:
            response = client.post(
                "/videos/compose",
                json={"topic": "job_missing_output"},
                headers=self._auth_headers(),
            )
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["message"], "job_not_found")


if __name__ == "__main__":
    unittest.main()
