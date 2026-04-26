from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from app.config import AppSettings
from app.errors import ServiceError
from app.services.frame_processor import PassthroughFrameProcessor
from app.services.kafka_service import KafkaService
from app.utils.contracts import validate_frame_request
from app.utils.crypto import decrypt_bytes, encrypt_bytes
from app.utils.job import derive_job_name_from_input_topic, derive_output_topic, normalize_group_id, validate_topic_name
from app.utils.runtime import RuntimeRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopicDispatchContext:
    topic: str
    job_name: str
    output_topic: str
    group_id: str


class TopicProcessService:
    def __init__(
        self,
        settings: AppSettings,
        kafka_service: KafkaService,
        runtime_registry: RuntimeRegistry,
        frame_processor: PassthroughFrameProcessor | None = None,
    ):
        self.settings = settings
        self.kafka_service = kafka_service
        self.runtime_registry = runtime_registry
        self.frame_processor = frame_processor or PassthroughFrameProcessor()

    def dispatch(self, topic: str, group_id: str | None = None) -> dict[str, Any]:
        normalized_topic, job_name, normalized_group_id = self._normalize_context(topic, group_id)

        context = TopicDispatchContext(
            topic=normalized_topic,
            job_name=job_name,
            output_topic=derive_output_topic(job_name),
            group_id=normalized_group_id,
        )

        summary, created = self.runtime_registry.register_or_reuse(
            topic=context.topic,
            job_name=context.job_name,
            output_topic=context.output_topic,
            group_id=context.group_id,
        )
        if created:
            self._start_background_task(context)

        return summary.dispatch_payload()

    def status(self, topic: str, group_id: str | None = None) -> dict[str, Any]:
        normalized_topic, _job_name, normalized_group_id = self._normalize_context(topic, group_id)
        summary = self.runtime_registry.snapshot(normalized_topic, normalized_group_id if group_id else None)
        if summary is None:
            raise ServiceError(
                status_code=404,
                error_code="task_not_found",
                message="No runtime task found for topic",
                details={"topic": normalized_topic, "group_id": normalized_group_id},
            )
        return summary.status_payload()

    @staticmethod
    def _normalize_context(topic: str, group_id: str | None) -> tuple[str, str, str]:
        normalized_topic = topic.strip()
        if not normalized_topic:
            raise ServiceError(status_code=400, error_code="invalid_topic", message="topic must not be empty")

        try:
            validate_topic_name(normalized_topic)
            job_name = derive_job_name_from_input_topic(normalized_topic)
            normalized_group_id = normalize_group_id(group_id, job_name=job_name)
        except ValueError as exc:
            raise ServiceError(status_code=400, error_code="invalid_topic", message=str(exc)) from exc

        return normalized_topic, job_name, normalized_group_id

    def _start_background_task(self, context: TopicDispatchContext) -> None:
        try:
            thread = threading.Thread(
                target=self._run_background_session,
                name=f"topic-process-{context.job_name}",
                args=(context,),
                daemon=True,
            )
            thread.start()
        except Exception as exc:
            self.runtime_registry.mark_failed(
                context.topic,
                context.group_id,
                error_code="background_task_start_failed",
                error_message=str(exc),
            )
            raise ServiceError(
                status_code=500,
                error_code="background_task_start_failed",
                message="Failed to start background topic worker",
            ) from exc

    def _run_background_session(self, context: TopicDispatchContext) -> None:
        self.runtime_registry.mark_running(context.topic, context.group_id)

        try:
            self._process_topic(context)
        except ServiceError as exc:
            logger.warning("Background session failed for %s: %s", context.topic, exc.message)
            self.runtime_registry.mark_failed(
                context.topic,
                context.group_id,
                error_code=exc.error_code,
                error_message=exc.message,
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Background session crashed for %s", context.topic)
            self.runtime_registry.mark_failed(
                context.topic,
                context.group_id,
                error_code="background_runtime_error",
                error_message=str(exc),
            )
        else:
            self.runtime_registry.mark_completed(context.topic, context.group_id)

    def _process_topic(self, context: TopicDispatchContext) -> None:
        self.kafka_service.ensure_topic(context.output_topic)

        consumer = self.kafka_service.create_frame_consumer(topic=context.topic, group_id=context.group_id)
        producer = self.kafka_service.create_frame_producer()

        started_at = time.monotonic()
        first_frame_deadline = started_at + self.settings.kafka.wait_first_frame_seconds
        last_valid_frame_at: float | None = None

        try:
            while True:
                now = time.monotonic()
                if now - started_at >= self.settings.kafka.max_process_seconds:
                    logger.info("Background session hit max duration for %s", context.topic)
                    return

                record = consumer.poll_record(timeout_seconds=0.5)
                if record is None:
                    now = time.monotonic()
                    if last_valid_frame_at is None:
                        if now >= first_frame_deadline:
                            raise ServiceError(
                                status_code=409,
                                error_code="first_frame_timeout",
                                message="No valid frame arrived before first-frame timeout",
                            )
                    elif now - last_valid_frame_at >= self.settings.kafka.idle_timeout_seconds:
                        logger.info("Background session idle timeout reached for %s", context.topic)
                        return
                    continue

                payload = self._decode_record_payload(record.value)
                if payload is None:
                    record.commit()
                    self.runtime_registry.record_diagnostic_error(
                        context.topic,
                        context.group_id,
                        error_code="invalid_frame_request",
                        error_message="Received non-JSON frame request payload",
                    )
                    continue

                if payload.get("job_name") != context.job_name:
                    record.commit()
                    continue

                last_valid_frame_at = time.monotonic()

                try:
                    validate_frame_request(payload)
                except ValueError as exc:
                    record.commit()
                    self.runtime_registry.record_diagnostic_error(
                        context.topic,
                        context.group_id,
                        error_code="invalid_frame_request",
                        error_message=str(exc),
                    )
                    continue

                last_valid_frame_at = time.monotonic()
                self.runtime_registry.increment_consumed(context.topic, context.group_id)
                frame_index = int(payload["frame_index"])

                try:
                    plain_frame = decrypt_bytes(
                        nonce_b64=payload["nonce_b64"],
                        ciphertext_b64=payload["ciphertext_b64"],
                        tag_b64=payload["tag_b64"],
                        aes_key=self.settings.aes_key_bytes,
                    )
                except Exception as exc:
                    error_payload = self._build_error_result_payload(
                        context.job_name,
                        frame_index,
                        error_code="FRAME_DECRYPT_FAILED",
                    )
                    producer.publish_frame_result(topic=context.output_topic, payload=error_payload)
                    record.commit()
                    self.runtime_registry.record_result(
                        context.topic,
                        context.group_id,
                        success=False,
                        error_code=str(error_payload["error_code"]),
                        error_message=str(exc),
                    )
                    continue

                try:
                    processed_frame = self.frame_processor.process(plain_frame, str(payload["content_type"]))
                except Exception as exc:
                    error_payload = self._build_error_result_payload(
                        context.job_name,
                        frame_index,
                        error_code="FRAME_PROCESS_FAILED",
                    )
                    producer.publish_frame_result(topic=context.output_topic, payload=error_payload)
                    record.commit()
                    self.runtime_registry.record_result(
                        context.topic,
                        context.group_id,
                        success=False,
                        error_code=str(error_payload["error_code"]),
                        error_message=str(exc),
                    )
                    continue

                try:
                    encrypted = encrypt_bytes(processed_frame, self.settings.aes_key_bytes)
                    result_payload = {
                        "job_name": context.job_name,
                        "frame_index": frame_index,
                        "status": "ok",
                        "nonce_b64": encrypted.nonce_b64,
                        "ciphertext_b64": encrypted.ciphertext_b64,
                        "tag_b64": encrypted.tag_b64,
                    }
                    producer.publish_frame_result(topic=context.output_topic, payload=result_payload)
                    record.commit()
                    self.runtime_registry.record_result(context.topic, context.group_id, success=True)
                except Exception as exc:
                    error_payload = self._build_error_result_payload(
                        context.job_name,
                        frame_index,
                        error_code="FRAME_ENCRYPT_FAILED",
                    )
                    producer.publish_frame_result(topic=context.output_topic, payload=error_payload)
                    record.commit()
                    self.runtime_registry.record_result(
                        context.topic,
                        context.group_id,
                        success=False,
                        error_code=str(error_payload["error_code"]),
                        error_message=str(exc),
                    )
        finally:
            producer.close()
            consumer.close()

    @staticmethod
    def _decode_record_payload(value: bytes) -> dict[str, Any] | None:
        try:
            decoded = value.decode("utf-8", errors="strict")
            payload = json.loads(decoded)
        except (UnicodeDecodeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _build_error_result_payload(self, job_name: str, frame_index: int, *, error_code: str) -> dict[str, Any]:
        error_marker = json.dumps({"error_code": error_code, "frame_index": frame_index}, ensure_ascii=False).encode(
            "utf-8"
        )
        encrypted = encrypt_bytes(error_marker, self.settings.aes_key_bytes)
        return {
            "job_name": job_name,
            "frame_index": frame_index,
            "status": "error",
            "nonce_b64": encrypted.nonce_b64,
            "ciphertext_b64": encrypted.ciphertext_b64,
            "tag_b64": encrypted.tag_b64,
            "error_code": error_code,
        }
