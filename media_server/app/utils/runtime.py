from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


ACTIVE_STATUSES = {"accepted", "running"}


@dataclass
class TaskSummary:
    topic: str
    job_name: str
    output_topic: str
    group_id: str
    status: str = "accepted"
    dispatch_mode: str = "async"
    consumed_count: int = 0
    published_count: int = 0
    success_count: int = 0
    error_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    completion_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def dispatch_payload(self) -> dict[str, str]:
        return {
            "job_name": self.job_name,
            "input_topic": self.topic,
            "output_topic": self.output_topic,
            "group_id": self.group_id,
            "dispatch_mode": self.dispatch_mode,
            "status": "accepted",
        }

    def status_payload(self) -> dict[str, object]:
        return {
            "topic": self.topic,
            "job_name": self.job_name,
            "output_topic": self.output_topic,
            "group_id": self.group_id,
            "dispatch_mode": self.dispatch_mode,
            "status": self.status,
            "consumed_count": self.consumed_count,
            "published_count": self.published_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
        }


class RuntimeRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: dict[tuple[str, str], TaskSummary] = {}

    @staticmethod
    def _key(topic: str, group_id: str) -> tuple[str, str]:
        return topic, group_id

    def register_or_reuse(
        self,
        *,
        topic: str,
        job_name: str,
        output_topic: str,
        group_id: str,
    ) -> tuple[TaskSummary, bool]:
        with self._lock:
            key = self._key(topic, group_id)
            existing = self._tasks.get(key)
            if existing is not None and existing.status in ACTIVE_STATUSES:
                return existing, False

            summary = TaskSummary(topic=topic, job_name=job_name, output_topic=output_topic, group_id=group_id)
            self._tasks[key] = summary
            return summary, True

    def mark_running(self, topic: str, group_id: str) -> None:
        with self._lock:
            summary = self._tasks[self._key(topic, group_id)]
            summary.status = "running"
            summary.updated_at = time.time()

    def increment_consumed(self, topic: str, group_id: str) -> None:
        with self._lock:
            summary = self._tasks[self._key(topic, group_id)]
            summary.consumed_count += 1
            summary.updated_at = time.time()

    def record_result(
        self,
        topic: str,
        group_id: str,
        *,
        success: bool,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._lock:
            summary = self._tasks[self._key(topic, group_id)]
            summary.published_count += 1
            if success:
                summary.success_count += 1
            else:
                summary.error_count += 1
                summary.error_code = error_code
                summary.error_message = error_message
            summary.updated_at = time.time()

    def record_diagnostic_error(self, topic: str, group_id: str, *, error_code: str, error_message: str) -> None:
        with self._lock:
            summary = self._tasks[self._key(topic, group_id)]
            summary.error_code = error_code
            summary.error_message = error_message
            summary.updated_at = time.time()

    def mark_failed(self, topic: str, group_id: str, *, error_code: str, error_message: str) -> None:
        with self._lock:
            summary = self._tasks[self._key(topic, group_id)]
            summary.status = "failed"
            summary.error_code = error_code
            summary.error_message = error_message
            summary.finished_at = time.time()
            summary.updated_at = summary.finished_at
            summary.completion_event.set()

    def mark_completed(self, topic: str, group_id: str) -> None:
        with self._lock:
            summary = self._tasks[self._key(topic, group_id)]
            summary.status = "completed"
            summary.finished_at = time.time()
            summary.updated_at = summary.finished_at
            summary.completion_event.set()

    def snapshot(self, topic: str, group_id: str | None = None) -> TaskSummary | None:
        with self._lock:
            if group_id is not None:
                return self._tasks.get(self._key(topic, group_id))

            candidates = [summary for (task_topic, _), summary in self._tasks.items() if task_topic == topic]
            if not candidates:
                return None
            return max(candidates, key=lambda summary: summary.updated_at)

    def wait_for_completion(self, topic: str, timeout: float, group_id: str | None = None) -> bool:
        summary = self.snapshot(topic, group_id)
        if summary is None:
            return False
        return summary.completion_event.wait(timeout)
