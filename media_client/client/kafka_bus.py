from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from client.contract import validate_kafka_publication
from client.config import KafkaSettings

try:
    from confluent_kafka import Consumer, KafkaError, Producer
    from confluent_kafka.admin import AdminClient
except Exception as import_exc:  # pragma: no cover - depends on runtime env
    Consumer = None
    KafkaError = None
    Producer = None
    AdminClient = None
    _KAFKA_IMPORT_ERROR = import_exc
else:  # pragma: no cover - import path branch
    _KAFKA_IMPORT_ERROR = None


@dataclass
class KafkaStageError(RuntimeError):
    code: str
    hint: str
    message: str

    def __str__(self) -> str:
        return self.message


def classify_kafka_error(exc: Exception) -> tuple[str, str]:
    message = str(exc).lower()

    if "authentication" in message or "sasl" in message or "invalid credentials" in message:
        return "auth_error", "Check KAFKA_SASL_USERNAME/KAFKA_SASL_PASSWORD and sasl mechanism"

    if "authorization" in message or "acl" in message or "not authorized" in message:
        return "acl_error", "Verify Kafka ACL for topics and consumer group"

    if "ssl" in message or "certificate" in message or "tls" in message:
        return "network_error", "Check KAFKA_SSL_CA_FILE and broker TLS certificate chain"

    if "timeout" in message or "timed out" in message or "transport" in message:
        return "network_error", "Check Kafka broker reachability and network policies"

    return "network_error", "Check Kafka connectivity and broker availability"


class KafkaDoctorClient:
    def __init__(self, settings: KafkaSettings):
        if _KAFKA_IMPORT_ERROR is not None:
            raise RuntimeError(
                "confluent-kafka is required for Kafka checks. Install project dependencies first."
            ) from _KAFKA_IMPORT_ERROR
        self.settings = settings

    def _common_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "bootstrap.servers": self.settings.bootstrap_servers,
            "security.protocol": self.settings.security_protocol,
            "sasl.mechanism": self.settings.sasl_mechanism,
        }

        if self.settings.security_protocol.startswith("SASL"):
            config["sasl.username"] = self.settings.sasl_username
            config["sasl.password"] = self.settings.sasl_password

        if self.settings.security_protocol in {"SSL", "SASL_SSL"}:
            config["ssl.ca.location"] = self.settings.ssl_ca_file

        return config

    def probe_metadata(self, timeout_seconds: float) -> dict[str, Any]:
        try:
            admin = AdminClient(self._common_config())
            metadata = admin.list_topics(timeout=timeout_seconds)
        except Exception as exc:  # pragma: no cover - integration dependent
            code, hint = classify_kafka_error(exc)
            raise KafkaStageError(code=code, hint=hint, message=str(exc)) from exc

        missing_topics = [name for name in self.settings.topics if name not in metadata.topics]
        if missing_topics:
            missing = ", ".join(missing_topics)
            raise KafkaStageError(
                code="acl_error",
                hint="Ensure topics exist and ACL grants metadata visibility",
                message=f"Missing or invisible topics: {missing}",
            )

        topic_errors = {}
        for topic_name in self.settings.topics:
            topic = metadata.topics[topic_name]
            if topic.error is not None:
                topic_errors[topic_name] = str(topic.error)

        if topic_errors:
            errors = ", ".join(f"{name}={error}" for name, error in sorted(topic_errors.items()))
            raise KafkaStageError(
                code="acl_error",
                hint="Verify topic-level ACL and metadata permissions",
                message=f"Topic metadata errors: {errors}",
            )

        return {
            "brokers": len(metadata.brokers),
            "topics_checked": self.settings.topics,
            "cluster_id": getattr(metadata, "cluster_id", None),
        }

    def probe_read_write(self, timeout_seconds: float) -> dict[str, Any]:
        probe_id = str(uuid.uuid4())
        topic = self.settings.topic_doctor_rw
        consumer_group = f"media-client-doctor-{uuid.uuid4()}"

        consumer_conf = {
            **self._common_config(),
            "group.id": consumer_group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        }

        producer_conf = {
            **self._common_config(),
            "acks": "all",
            "enable.idempotence": True,
        }

        producer = Producer(producer_conf)
        consumer = Consumer(consumer_conf)
        delivery_errors: list[str] = []

        def _delivery_callback(err: Any, _msg: Any) -> None:
            if err is not None:
                delivery_errors.append(str(err))

        payload = {
            "probe_id": probe_id,
            "created_at_ms": int(time.time() * 1000),
            "source": "media_client_doctor",
        }
        validate_kafka_publication(payload)

        try:
            consumer.subscribe([topic])
            assignment_deadline = time.monotonic() + min(5.0, timeout_seconds * 0.5)
            while time.monotonic() < assignment_deadline:
                message = consumer.poll(0.2)
                if message is None:
                    if consumer.assignment():
                        break
                    continue

                if message.error() is not None:
                    if KafkaError is not None and message.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    text = str(message.error())
                    code, hint = classify_kafka_error(RuntimeError(text))
                    raise KafkaStageError(code=code, hint=hint, message=text)

            if not consumer.assignment():
                raise KafkaStageError(
                    code="network_error",
                    hint="Check consumer group join latency and broker connectivity",
                    message=f"Kafka consumer assignment timed out for topic '{topic}'",
                )

            producer.produce(
                topic=topic,
                key=probe_id.encode("utf-8"),
                value=json.dumps(payload).encode("utf-8"),
                on_delivery=_delivery_callback,
            )
            producer.flush(timeout=timeout_seconds)

            if delivery_errors:
                text = "; ".join(delivery_errors)
                code, hint = classify_kafka_error(RuntimeError(text))
                raise KafkaStageError(code=code, hint=hint, message=text)

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                message = consumer.poll(0.5)
                if message is None:
                    continue

                if message.error() is not None:
                    if KafkaError is not None and message.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    text = str(message.error())
                    code, hint = classify_kafka_error(RuntimeError(text))
                    raise KafkaStageError(code=code, hint=hint, message=text)

                key = message.key().decode("utf-8", errors="replace") if message.key() else ""
                if key == probe_id:
                    return {
                        "topic": topic,
                        "probe_id": probe_id,
                        "consumer_group": consumer_group,
                    }

                try:
                    decoded = json.loads(message.value().decode("utf-8", errors="replace"))
                except (ValueError, AttributeError):
                    continue

                if decoded.get("probe_id") == probe_id:
                    return {
                        "topic": topic,
                        "probe_id": probe_id,
                        "consumer_group": consumer_group,
                    }

            raise KafkaStageError(
                code="rw_timeout",
                hint="Verify producer/consumer ACL and topic propagation latency",
                message=f"Kafka read/write probe timed out for topic '{topic}'",
            )

        except KafkaStageError:
            raise
        except Exception as exc:  # pragma: no cover - integration dependent
            code, hint = classify_kafka_error(exc)
            raise KafkaStageError(code=code, hint=hint, message=str(exc)) from exc
        finally:
            consumer.close()
