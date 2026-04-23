from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from main import create_app

TEST_API_KEY = "openai-test-token-12345"


def _write_reference_audio(path: Path, sample_rate: int = 22050) -> None:
    sample_count = int(sample_rate * 1.0)
    timeline = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    waveform = 0.22 * np.sin(2.0 * np.pi * 210.0 * timeline)
    sf.write(str(path), waveform, sample_rate)


class OpenAiCompatibleApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(
            os.environ,
            {
                "AUTH_REQUIRED": "true",
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

    def _speech_payload(self) -> dict[str, object]:
        return {
            "model": "tts-1",
            "input": "speech payload",
            "voice": "alloy",
            "response_format": "wav",
            "stream": False,
        }

    def test_models_endpoint_returns_minimal_model_list(self) -> None:
        response = self.client.get("/v1/models", headers=self.auth_headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "list")
        model_ids = [item["id"] for item in payload["data"]]
        self.assertIn("tts-1", model_ids)
        self.assertIn("tts-1-hd", model_ids)
        self.assertIn("moss-tts-nano-onnx", model_ids)

    def test_models_endpoint_rejects_missing_or_invalid_bearer_token(self) -> None:
        no_token = self.client.get("/v1/models")
        self.assertEqual(no_token.status_code, 401)
        self.assertEqual(no_token.headers.get("www-authenticate"), "Bearer")

        invalid_token = self.client.get(
            "/v1/models",
            headers={"Authorization": "Bearer invalid-token"},
        )
        self.assertEqual(invalid_token.status_code, 401)
        self.assertEqual(invalid_token.headers.get("www-authenticate"), "Bearer")

    def test_voice_listing_exposes_alias_to_canonical_mapping(self) -> None:
        response = self.client.get("/v1/audio/voices", headers=self.auth_headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "list")
        self.assertGreater(len(payload["data"]), 0)

        adam = next((item for item in payload["data"] if item["canonical"] == "Adam"), None)
        self.assertIsNotNone(adam)
        self.assertIn("alloy", adam["aliases"])
        self.assertIn("onyx", adam["aliases"])
        self.assertTrue(adam["prompt_audio_configured"])
        self.assertTrue(adam["prompt_audio_exists"])

    def test_voice_listing_rejects_missing_bearer_token(self) -> None:
        response = self.client.get("/v1/audio/voices")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get("www-authenticate"), "Bearer")

    def test_speech_stream_mode_returns_non_empty_chunks(self) -> None:
        payload = {
            "model": "tts-1",
            "input": "stream mode synthesis from openai compatible endpoint",
            "voice": "alloy",
            "response_format": "pcm",
            "stream": True,
        }

        with self.client.stream(
            "POST",
            "/v1/audio/speech",
            json=payload,
            headers=self.auth_headers,
        ) as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("audio/pcm", response.headers.get("content-type", ""))
            chunks = list(response.iter_bytes())

        self.assertGreater(len(chunks), 0)
        self.assertGreater(len(b"".join(chunks)), 0)

    def test_speech_non_stream_supports_wav_and_pcm_formats(self) -> None:
        wav_payload = {
            "model": "tts-1",
            "input": "wav format test",
            "voice": "alloy",
            "response_format": "wav",
            "stream": False,
        }
        pcm_payload = {
            "model": "tts-1",
            "input": "pcm format test",
            "voice": "alloy",
            "response_format": "pcm",
            "stream": False,
        }

        wav_response = self.client.post("/v1/audio/speech", json=wav_payload, headers=self.auth_headers)
        self.assertEqual(wav_response.status_code, 200)
        self.assertIn("audio/wav", wav_response.headers.get("content-type", ""))
        self.assertGreater(len(wav_response.content), 0)
        self.assertTrue(wav_response.content.startswith(b"RIFF"))
        self.assertEqual(wav_response.content[8:12], b"WAVE")

        pcm_response = self.client.post("/v1/audio/speech", json=pcm_payload, headers=self.auth_headers)
        self.assertEqual(pcm_response.status_code, 200)
        self.assertIn("audio/pcm", pcm_response.headers.get("content-type", ""))
        self.assertGreater(len(pcm_response.content), 0)

    def test_speech_unauthorized_is_rejected_before_synthesis_logic(self) -> None:
        with patch("app.services.openai_speech_service.OpenAiSpeechService.synthesize_audio") as synthesize:
            response = self.client.post("/v1/audio/speech", json=self._speech_payload())

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get("www-authenticate"), "Bearer")
        synthesize.assert_not_called()

    def test_voice_resolution_supports_builtin_and_reference_with_reference_priority(self) -> None:
        builtin_payload = {
            "model": "tts-1",
            "input": "voice fallback without reference",
            "voice": "alloy",
            "response_format": "wav",
            "stream": False,
        }
        builtin_response = self.client.post("/v1/audio/speech", json=builtin_payload, headers=self.auth_headers)
        self.assertEqual(builtin_response.status_code, 200)
        self.assertGreater(len(builtin_response.content), 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.wav"
            _write_reference_audio(reference_path)

            with_reference_alloy = {
                "model": "tts-1",
                "input": "reference audio should override voice",
                "voice": "alloy",
                "response_format": "wav",
                "stream": False,
                "reference_audio_path": str(reference_path),
            }
            with_reference_echo = {
                "model": "tts-1",
                "input": "reference audio should override voice",
                "voice": "echo",
                "response_format": "wav",
                "stream": False,
                "reference_audio_path": str(reference_path),
            }

            alloy_response = self.client.post("/v1/audio/speech", json=with_reference_alloy, headers=self.auth_headers)
            echo_response = self.client.post("/v1/audio/speech", json=with_reference_echo, headers=self.auth_headers)
            self.assertEqual(alloy_response.status_code, 200)
            self.assertEqual(echo_response.status_code, 200)
            self.assertGreater(len(alloy_response.content), 0)
            self.assertEqual(alloy_response.content, echo_response.content)
            self.assertNotEqual(builtin_response.content, alloy_response.content)

    def test_voice_resolution_accepts_openai_alias_and_moss_canonical_voice(self) -> None:
        alias_payload = {
            "model": "tts-1",
            "input": "mapping check for canonical voices",
            "voice": "alloy",
            "response_format": "wav",
            "stream": False,
        }
        canonical_payload = {
            "model": "tts-1",
            "input": "mapping check for canonical voices",
            "voice": "Adam",
            "response_format": "wav",
            "stream": False,
        }

        alias_response = self.client.post("/v1/audio/speech", json=alias_payload, headers=self.auth_headers)
        canonical_response = self.client.post(
            "/v1/audio/speech",
            json=canonical_payload,
            headers=self.auth_headers,
        )
        self.assertEqual(alias_response.status_code, 200)
        self.assertEqual(canonical_response.status_code, 200)
        self.assertGreater(len(alias_response.content), 0)
        self.assertEqual(alias_response.content, canonical_response.content)

    def test_invalid_model_empty_input_and_unsupported_format_return_400(self) -> None:
        invalid_model = self.client.post(
            "/v1/audio/speech",
            json={
                "model": "unknown-model",
                "input": "hello",
                "voice": "alloy",
                "response_format": "wav",
                "stream": False,
            },
            headers=self.auth_headers,
        )
        self.assertEqual(invalid_model.status_code, 400)

        empty_input = self.client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": " ",
                "voice": "alloy",
                "response_format": "wav",
                "stream": False,
            },
            headers=self.auth_headers,
        )
        self.assertEqual(empty_input.status_code, 400)

        unsupported_format = self.client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": "hello",
                "voice": "alloy",
                "response_format": "mp3",
                "stream": False,
            },
            headers=self.auth_headers,
        )
        self.assertEqual(unsupported_format.status_code, 400)

    def test_runtime_failure_is_mapped_to_500(self) -> None:
        with patch(
            "app.services.openai_speech_service.OpenAiSpeechService.synthesize_audio",
            side_effect=RuntimeError("runtime crashed"),
        ):
            response = self.client.post(
                "/v1/audio/speech",
                json={
                    "model": "tts-1",
                    "input": "trigger backend failure",
                    "voice": "alloy",
                    "response_format": "wav",
                    "stream": False,
                },
                headers=self.auth_headers,
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "speech synthesis failed")

    def test_openai_sdk_style_base_url_invocation(self) -> None:
        # OpenAI SDK style: set base_url to /v1 and call /audio/speech.
        with TestClient(create_app(), base_url="http://testserver/v1") as sdk_style_client:
            response = sdk_style_client.post(
                "/audio/speech",
                json={
                    "model": "tts-1",
                    "input": "sdk-style invocation",
                    "voice": "alloy",
                    "response_format": "wav",
                },
                headers=self.auth_headers,
            )
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.content), 0)


if __name__ == "__main__":
    unittest.main()
