from __future__ import annotations

import re
import secrets
import string
import time

JOB_NAME_PATTERN = re.compile(r"^job_[0-9]{13}_[a-z0-9]{8}$")
TOPIC_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")


def generate_job_name() -> str:
    utc_millis = int(time.time() * 1000)
    random_suffix = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    return f"job_{utc_millis}_{random_suffix}"


def validate_job_name(job_name: str) -> None:
    if not JOB_NAME_PATTERN.match(job_name):
        raise ValueError("job_name does not match required format")


def derive_input_topic(job_name: str) -> str:
    validate_job_name(job_name)
    return f"{job_name}_input"


def derive_job_name_from_topic(topic: str) -> str:
    normalized = topic.strip()
    if not normalized:
        raise ValueError("topic is required")

    if normalized.endswith("_output"):
        return normalized[: -len("_output")]

    if normalized.endswith("_input"):
        return normalized[: -len("_input")]

    return normalized


def validate_topic_name(topic: str) -> None:
    if not topic:
        raise ValueError("topic must not be empty")
    if not TOPIC_NAME_PATTERN.match(topic):
        raise ValueError("topic contains invalid characters")
