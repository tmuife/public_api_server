from __future__ import annotations

import base64
import unittest

from app.utils.contracts import validate_frame_request, validate_frame_result, validate_kafka_publication
from app.utils.crypto import decrypt_bytes, encrypt_bytes
from app.utils.job import build_group_id, derive_job_name_from_input_topic, derive_output_topic, normalize_group_id


class CryptoContractTests(unittest.TestCase):
    def test_encrypt_decrypt_round_trip(self) -> None:
        key = b"k" * 32
        plaintext = b"frame-binary-data"

        encrypted = encrypt_bytes(plaintext, key)
        decrypted = decrypt_bytes(
            nonce_b64=encrypted.nonce_b64,
            ciphertext_b64=encrypted.ciphertext_b64,
            tag_b64=encrypted.tag_b64,
            aes_key=key,
        )

        self.assertEqual(decrypted, plaintext)

    def test_frame_request_contract_accepts_required_fields(self) -> None:
        payload = {
            "job_name": "job_1776945123456_ab12cd34",
            "frame_index": 0,
            "nonce_b64": base64.b64encode(b"1" * 12).decode("utf-8"),
            "ciphertext_b64": base64.b64encode(b"cipher").decode("utf-8"),
            "tag_b64": base64.b64encode(b"2" * 16).decode("utf-8"),
            "content_type": "image/jpeg",
        }
        validate_frame_request(payload)

    def test_frame_result_error_requires_error_code(self) -> None:
        payload = {
            "job_name": "job_1776945123456_ab12cd34",
            "frame_index": 0,
            "status": "error",
            "nonce_b64": base64.b64encode(b"1" * 12).decode("utf-8"),
            "ciphertext_b64": base64.b64encode(b"cipher").decode("utf-8"),
            "tag_b64": base64.b64encode(b"2" * 16).decode("utf-8"),
        }
        with self.assertRaisesRegex(ValueError, "error_code"):
            validate_frame_result(payload)

    def test_publication_rejects_sensitive_field_names(self) -> None:
        payload = {
            "job_name": "job_1776945123456_ab12cd34",
            "frame_index": 0,
            "nonce_b64": base64.b64encode(b"1" * 12).decode("utf-8"),
            "ciphertext_b64": base64.b64encode(b"cipher").decode("utf-8"),
            "tag_b64": base64.b64encode(b"2" * 16).decode("utf-8"),
            "content_type": "image/jpeg",
        }
        with self.assertRaisesRegex(ValueError, "Kafka headers"):
            validate_kafka_publication(payload, headers={"x-api-token": "secret"})

    def test_topic_and_group_derivation(self) -> None:
        topic = "job_1776945123456_ab12cd34_input"
        job_name = derive_job_name_from_input_topic(topic)
        self.assertEqual(job_name, "job_1776945123456_ab12cd34")
        self.assertEqual(derive_output_topic(job_name), "job_1776945123456_ab12cd34_output")
        self.assertEqual(build_group_id(job_name), "media-server-job_1776945123456_ab12cd34")
        self.assertEqual(normalize_group_id("debug-run-001", job_name=job_name), "debug-run-001")

    def test_invalid_group_id_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "group_id"):
            normalize_group_id("bad group", job_name="job_1776945123456_ab12cd34")
