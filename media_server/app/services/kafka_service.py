from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from app.config import KafkaSettings
from app.errors import ServiceError
from app.utils.contracts import validate_frame_result, validate_kafka_publication
from app.utils.job import validate_topic_name

logger = logging.getLogger(__name__)


@dataclass
class ConsumedRecord:
    value: bytes
    commit: Callable[[], None]


class KafkaFrameConsumer:
    def __init__(self, consumer: Any, kafka_error: Any):
        self._consumer = consumer
        self._kafka_error = kafka_error

    def poll_record(self, timeout_seconds: float) -> ConsumedRecord | None:
        message = self._consumer.poll(timeout_seconds)
        if message is None:
            return None

        if message.error() is not None:
            if self._kafka_error is not None and message.error().code() == self._kafka_error._PARTITION_EOF:
                return None
            raise ServiceError(
                status_code=500,
                error_code="kafka_consume_failed",
                message="Kafka consume error",
                details={"reason": str(message.error())},
            )

        def _commit() -> None:
            self._consumer.commit(message=message, asynchronous=False)

        return ConsumedRecord(value=message.value(), commit=_commit)

    def close(self) -> None:
        self._consumer.close()


class KafkaFrameProducer:
    def __init__(self, producer: Any, timeout_seconds: float):
        self._producer = producer
        self._timeout_seconds = timeout_seconds

    def publish_frame_result(self, *, topic: str, payload: dict[str, Any]) -> None:
        validate_topic_name(topic)
        validate_frame_result(payload)
        validate_kafka_publication(payload)

        delivery_errors: list[str] = []

        def _delivery_callback(err: Any, _msg: Any) -> None:
            if err is not None:
                delivery_errors.append(str(err))

        key = f"{payload['job_name']}:{payload['frame_index']}".encode("utf-8")
        self._producer.produce(
            topic=topic,
            key=key,
            value=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            on_delivery=_delivery_callback,
        )
        self._producer.flush(timeout=self._timeout_seconds)

        if delivery_errors:
            raise ServiceError(
                status_code=500,
                error_code="kafka_publish_failed",
                message="Failed to publish result frame",
                details={"errors": delivery_errors[:3]},
            )

    def close(self) -> None:
        try:
            self._producer.flush(timeout=self._timeout_seconds)
        except Exception:
            logger.debug("Producer flush during close failed", exc_info=True)


class KafkaService:
    def __init__(self, settings: KafkaSettings):
        self.settings = settings

    @staticmethod
    def _load_kafka_modules() -> tuple[Any, Any, Any, Any, Any]:
        try:
            from confluent_kafka import Consumer, KafkaError, Producer
            from confluent_kafka.admin import AdminClient, NewTopic
        except Exception as exc:  # pragma: no cover
            raise ServiceError(
                status_code=500,
                error_code="kafka_dependency_missing",
                message="confluent-kafka is required for Kafka operations",
            ) from exc

        return Consumer, KafkaError, Producer, AdminClient, NewTopic

    def ensure_topic(self, topic: str) -> None:
        validate_topic_name(topic)
        _Consumer, _KafkaError, _Producer, AdminClient, NewTopic = self._load_kafka_modules()

        admin = AdminClient(self.settings.as_admin_config())
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

    def create_frame_consumer(self, *, topic: str, group_id: str) -> KafkaFrameConsumer:
        validate_topic_name(topic)
        Consumer, KafkaError, _Producer, _AdminClient, _NewTopic = self._load_kafka_modules()
        config = {
            **self.settings.as_consumer_config(),
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
        consumer = Consumer(config)
        consumer.subscribe([topic])
        return KafkaFrameConsumer(consumer, KafkaError)

    def create_frame_producer(self) -> KafkaFrameProducer:
        _Consumer, _KafkaError, Producer, _AdminClient, _NewTopic = self._load_kafka_modules()
        config = {
            **self.settings.as_producer_config(),
            "acks": "all",
            "enable.idempotence": True,
        }
        return KafkaFrameProducer(Producer(config), timeout_seconds=self.settings.produce_timeout_seconds)
