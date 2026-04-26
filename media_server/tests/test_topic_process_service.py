from __future__ import annotations

import base64
import json
import time
import unittest

from app.config import AppSettings, KafkaSettings
from app.services.topic_process_service import TopicProcessService
from app.utils.crypto import encrypt_bytes
from app.utils.runtime import RuntimeRegistry


class FakeRecord:
    def __init__(self, value: bytes):
        self.value = value
        self.committed = False

    def commit(self) -> None:
        self.committed = True


class FakeConsumer:
    def __init__(self, records: list[FakeRecord]):
        self.records = records
        self.index = 0
        self.closed = False

    def poll_record(self, timeout_seconds: float):
        _ = timeout_seconds
        if self.index >= len(self.records):
            return None
        record = self.records[self.index]
        self.index += 1
        return record

    def close(self) -> None:
        self.closed = True


class FakeProducer:
    def __init__(self):
        self.published: list[tuple[str, dict]] = []
        self.closed = False

    def publish_frame_result(self, *, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))

    def close(self) -> None:
        self.closed = True


class FakeKafkaService:
    def __init__(self, records: list[FakeRecord]):
        self.records = records
        self.ensure_topic_calls: list[str] = []
        self.consumer_calls = 0
        self.consumer_group_ids: list[str] = []
        self.last_producer = FakeProducer()

    def ensure_topic(self, topic: str) -> None:
        self.ensure_topic_calls.append(topic)

    def create_frame_consumer(self, *, topic: str, group_id: str):
        _ = topic
        self.consumer_group_ids.append(group_id)
        self.consumer_calls += 1
        return FakeConsumer(self.records)

    def create_frame_producer(self):
        return self.last_producer


class SlowPassthroughProcessor:
    def __init__(self):
        self.started = False
        self.enter_event = None

    def process(self, frame_bytes: bytes, content_type: str) -> bytes:
        _ = content_type
        self.started = True
        time.sleep(0.05)
        return frame_bytes


class FailingProcessor:
    def process(self, frame_bytes: bytes, content_type: str) -> bytes:
        _ = frame_bytes
        _ = content_type
        raise RuntimeError("processor exploded")


def build_settings() -> AppSettings:
    key = b"a" * 32
    return AppSettings(
        media_api_access_token="token",
        aes_key_b64=base64.b64encode(key).decode("utf-8"),
        aes_key_bytes=key,
        kafka=KafkaSettings(
            bootstrap_servers="localhost:9092",
            security_protocol="PLAINTEXT",
            sasl_mechanism=None,
            ssl_ca_file=None,
            produce_timeout_seconds=1.0,
            wait_first_frame_seconds=1.0,
            idle_timeout_seconds=0.2,
            max_process_seconds=2.0,
            topic_partitions=1,
            topic_replication_factor=1,
        ),
    )


def build_encrypted_request_payload(settings: AppSettings, frame_index: int = 0) -> dict[str, object]:
    encrypted = encrypt_bytes(b"frame-bytes", settings.aes_key_bytes)
    return {
        "job_name": "job_1776945123456_ab12cd34",
        "frame_index": frame_index,
        "nonce_b64": encrypted.nonce_b64,
        "ciphertext_b64": encrypted.ciphertext_b64,
        "tag_b64": encrypted.tag_b64,
        "content_type": "image/jpeg",
    }


class TopicProcessServiceTests(unittest.TestCase):
    def test_duplicate_dispatch_reuses_active_task(self) -> None:
        settings = build_settings()
        payload = build_encrypted_request_payload(settings)
        record = FakeRecord(json.dumps(payload).encode("utf-8"))
        kafka_service = FakeKafkaService([record])
        runtime_registry = RuntimeRegistry()
        service = TopicProcessService(
            settings=settings,
            kafka_service=kafka_service,
            runtime_registry=runtime_registry,
            frame_processor=SlowPassthroughProcessor(),
        )

        first = service.dispatch("job_1776945123456_ab12cd34_input")
        second = service.dispatch("job_1776945123456_ab12cd34_input")

        self.assertEqual(first["group_id"], second["group_id"])
        self.assertTrue(runtime_registry.wait_for_completion("job_1776945123456_ab12cd34_input", 2.0))
        self.assertEqual(kafka_service.consumer_calls, 1)

    def test_custom_group_id_starts_independent_task(self) -> None:
        settings = build_settings()
        payload = build_encrypted_request_payload(settings)
        records = [
            FakeRecord(json.dumps(payload).encode("utf-8")),
            FakeRecord(json.dumps(payload).encode("utf-8")),
        ]
        kafka_service = FakeKafkaService(records)
        runtime_registry = RuntimeRegistry()
        service = TopicProcessService(
            settings=settings,
            kafka_service=kafka_service,
            runtime_registry=runtime_registry,
            frame_processor=SlowPassthroughProcessor(),
        )

        default_dispatch = service.dispatch("job_1776945123456_ab12cd34_input")
        custom_dispatch = service.dispatch("job_1776945123456_ab12cd34_input", "debug-run-001")

        self.assertEqual(default_dispatch["group_id"], "media-server-job_1776945123456_ab12cd34")
        self.assertEqual(custom_dispatch["group_id"], "debug-run-001")
        self.assertTrue(
            runtime_registry.wait_for_completion(
                "job_1776945123456_ab12cd34_input",
                2.0,
                group_id="media-server-job_1776945123456_ab12cd34",
            )
        )
        self.assertTrue(
            runtime_registry.wait_for_completion(
                "job_1776945123456_ab12cd34_input",
                2.0,
                group_id="debug-run-001",
            )
        )
        self.assertEqual(kafka_service.consumer_calls, 2)
        self.assertIn("debug-run-001", kafka_service.consumer_group_ids)

    def test_status_returns_runtime_summary(self) -> None:
        settings = build_settings()
        payload = build_encrypted_request_payload(settings)
        record = FakeRecord(json.dumps(payload).encode("utf-8"))
        kafka_service = FakeKafkaService([record])
        runtime_registry = RuntimeRegistry()
        service = TopicProcessService(
            settings=settings,
            kafka_service=kafka_service,
            runtime_registry=runtime_registry,
            frame_processor=SlowPassthroughProcessor(),
        )

        service.dispatch("job_1776945123456_ab12cd34_input", "debug-run-001")
        self.assertTrue(
            runtime_registry.wait_for_completion(
                "job_1776945123456_ab12cd34_input",
                2.0,
                group_id="debug-run-001",
            )
        )

        status = service.status("job_1776945123456_ab12cd34_input", "debug-run-001")

        self.assertEqual(status["group_id"], "debug-run-001")
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["published_count"], 1)

    def test_worker_publishes_error_result_when_processor_fails(self) -> None:
        settings = build_settings()
        payload = build_encrypted_request_payload(settings)
        record = FakeRecord(json.dumps(payload).encode("utf-8"))
        kafka_service = FakeKafkaService([record])
        runtime_registry = RuntimeRegistry()
        service = TopicProcessService(
            settings=settings,
            kafka_service=kafka_service,
            runtime_registry=runtime_registry,
            frame_processor=FailingProcessor(),
        )

        service.dispatch("job_1776945123456_ab12cd34_input")
        self.assertTrue(runtime_registry.wait_for_completion("job_1776945123456_ab12cd34_input", 2.0))

        self.assertEqual(len(kafka_service.last_producer.published), 1)
        topic, result_payload = kafka_service.last_producer.published[0]
        self.assertEqual(topic, "job_1776945123456_ab12cd34_output")
        self.assertEqual(result_payload["status"], "error")
        self.assertEqual(result_payload["error_code"], "FRAME_PROCESS_FAILED")
        self.assertTrue(record.committed)
