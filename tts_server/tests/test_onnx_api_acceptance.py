from __future__ import annotations

import unittest

from tests.acceptance_helpers import build_auth_headers, build_speech_payload, build_test_client


class OnnxApiAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = build_test_client()

    def tearDown(self) -> None:
        self.client.close()

    def test_stream_success_returns_audio_chunks(self) -> None:
        payload = build_speech_payload(
            input_text="onnx acceptance stream check",
            response_format="pcm",
            stream=True,
        )
        headers = build_auth_headers()

        with self.client.stream("POST", "/audio/speech", json=payload, headers=headers) as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("audio/pcm", response.headers.get("content-type", ""))
            chunks = list(response.iter_bytes())

        self.assertGreater(len(chunks), 0, "Expected at least one streamed chunk")
        self.assertGreater(len(b"".join(chunks)), 0, "Expected streamed chunks to contain audio bytes")

    def test_non_stream_success_returns_full_payload(self) -> None:
        payload = build_speech_payload(
            input_text="onnx acceptance non-stream check",
            response_format="wav",
            stream=False,
        )
        headers = build_auth_headers()

        response = self.client.post("/audio/speech", json=payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("audio/wav", response.headers.get("content-type", ""))
        self.assertGreater(len(response.content), 0)
        self.assertTrue(response.content.startswith(b"RIFF"))
        self.assertEqual(response.content[8:12], b"WAVE")

    def test_invalid_model_returns_4xx_with_actionable_message(self) -> None:
        payload = build_speech_payload(model="invalid-model-for-acceptance", input_text="invalid model path")
        response = self.client.post("/audio/speech", json=payload, headers=build_auth_headers())

        self.assertEqual(response.status_code, 400)
        detail = str(response.json().get("detail", "")).lower()
        self.assertIn("unsupported model", detail)

    def test_empty_input_returns_4xx_with_actionable_message(self) -> None:
        payload = build_speech_payload(input_text="   ")
        response = self.client.post("/audio/speech", json=payload, headers=build_auth_headers())

        self.assertEqual(response.status_code, 400)
        detail = str(response.json().get("detail", "")).lower()
        self.assertIn("input cannot be empty", detail)


if __name__ == "__main__":
    unittest.main()
