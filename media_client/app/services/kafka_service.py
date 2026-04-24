from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Mapping

from app.config import KafkaSettings
from app.errors import ServiceError
from app.utils.contracts import validate_frame_request, validate_frame_result, validate_kafka_publication
from app.utils.job import validate_topic_name

logger = logging.getLogger(__name__)


class KafkaService:
    def __init__(self, settings: KafkaSettings):
        self.settings = settings

    @staticmethod
    def _load_kafka_modules() -> tuple[Any, Any, Any, Any, Any]:
        try:
            from confluent_kafka import Consumer, KafkaError, Producer
            from confluent_kafka.admin import AdminClient, NewTopic
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            raise ServiceError(
                status_code=500,
                error_code="kafka_dependency_missing",
                message="confluent-kafka is required for Kafka operations",
            ) from exc

        return Consumer, KafkaError, Producer, AdminClient, NewTopic

    def _base_config(self) -> dict[str, Any]:
        return self.settings.as_client_config()

    def ensure_topic(self, topic: str) -> None:
        validate_topic_name(topic)
        _Consumer, _KafkaError, _Producer, AdminClient, NewTopic = self._load_kafka_modules()

        admin = AdminClient(self._base_config())
        try:
            metadata = admin.list_topics(topic=topic, timeout=10)
            topic_meta = metadata.topics.get(topic)
            if topic_meta is not None and topic_meta.error is None:
                return

            futures = admin.create_topics(
                [
                    NewTopic(
                        topic=topic,
                        num_partitions=self.settings.topic_partitions,
                        replication_factor=self.settings.topic_replication_factor,
                    )
                ]
            )
            futures[topic].result(timeout=10)
        except Exception as exc:
            text = str(exc).lower()
            if "topic already exists" in text or "already exists" in text:
                return
            raise ServiceError(
                status_code=409,
                error_code="topic_create_failed",
                message=f"Failed to create or verify topic: {topic}",
                details={"reason": str(exc)},
            ) from exc

    def publish_frame_requests(self, *, topic: str, messages: list[Mapping[str, Any]]) -> None:
        validate_topic_name(topic)
        _Consumer, _KafkaError, Producer, _AdminClient, _NewTopic = self._load_kafka_modules()

        producer_config = {
            **self._base_config(),
            "acks": "all",
            "enable.idempotence": True,
        }

        producer = Producer(producer_config)
        delivery_errors: list[str] = []

        def _delivery_callback(err: Any, _msg: Any) -> None:
            if err is not None:
                delivery_errors.append(str(err))

        for payload in messages:
            validate_frame_request(payload)
            validate_kafka_publication(payload)
            key = f"{payload['job_name']}:{payload['frame_index']}".encode("utf-8")
            producer.produce(
                topic=topic,
                key=key,
                value=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                on_delivery=_delivery_callback,
            )

        producer.flush(timeout=self.settings.produce_timeout_seconds)

        if delivery_errors:
            raise ServiceError(
                status_code=500,
                error_code="kafka_publish_failed",
                message="One or more frame messages failed to publish",
                details={"errors": delivery_errors[:3]},
            )

    def consume_frame_results(
        self,
        *,
        topic: str,
        job_name: str,
        expected_count: int,
        timeout_seconds: float,
    ) -> dict[int, dict[str, Any]]:
        validate_topic_name(topic)
        Consumer, KafkaError, _Producer, _AdminClient, _NewTopic = self._load_kafka_modules()

        consumer_config = {
            **self._base_config(),
            "group.id": f"media-client-compose-{uuid.uuid4()}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        consumer = Consumer(consumer_config)
        consumer.subscribe([topic])

        frames_by_index: dict[int, dict[str, Any]] = {}
        deadline = time.monotonic() + timeout_seconds

        try:
            while time.monotonic() < deadline and len(frames_by_index) < expected_count:
                message = consumer.poll(0.5)
                if message is None:
                    continue

                if message.error() is not None:
                    if KafkaError is not None and message.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise ServiceError(
                        status_code=500,
                        error_code="kafka_consume_failed",
                        message="Kafka consume error",
                        details={"reason": str(message.error())},
                    )

                try:
                    payload = json.loads(message.value().decode("utf-8", errors="replace"))
                except ValueError as exc:
                    raise ServiceError(
                        status_code=409,
                        error_code="invalid_frame_result",
                        message="Received non-JSON frame result payload",
                    ) from exc

                if payload.get("job_name") != job_name:
                    continue

                try:
                    validate_frame_result(payload)
                except ValueError as exc:
                    raise ServiceError(
                        status_code=409,
                        error_code="invalid_frame_result",
                        message=str(exc),
                    ) from exc

                frame_index = int(payload["frame_index"])

                if payload.get("status") == "error":
                    raise ServiceError(
                        status_code=409,
                        error_code="downstream_frame_error",
                        message="Downstream worker reported frame processing error",
                        details={
                            "frame_index": frame_index,
                            "error_code": payload.get("error_code"),
                        },
                    )

                frames_by_index.setdefault(frame_index, payload)

            if len(frames_by_index) < expected_count:
                missing = [index for index in range(expected_count) if index not in frames_by_index]
                raise ServiceError(
                    status_code=409,
                    error_code="compose_timeout",
                    message="Timed out while waiting for expected frame results",
                    details={
                        "missing_count": len(missing),
                        "missing_frame_indexes": missing[:50],
                    },
                )

            return frames_by_index
        finally:
            consumer.close()
