from __future__ import annotations

import re

JOB_NAME_PATTERN = re.compile(r"^job_[0-9]{13}_[a-z0-9]{8}$")
TOPIC_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")
GROUP_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")
INPUT_TOPIC_SUFFIX = "_input"
OUTPUT_TOPIC_SUFFIX = "_output"


def validate_job_name(job_name: str) -> None:
    if not JOB_NAME_PATTERN.match(job_name):
        raise ValueError("job_name does not match required format")


def validate_topic_name(topic: str) -> None:
    if not topic:
        raise ValueError("topic must not be empty")
    if not TOPIC_NAME_PATTERN.match(topic):
        raise ValueError("topic contains invalid characters")


def derive_job_name_from_input_topic(topic: str) -> str:
    normalized = topic.strip()
    validate_topic_name(normalized)
    if not normalized.endswith(INPUT_TOPIC_SUFFIX):
        raise ValueError("topic must end with '_input'")
    job_name = normalized[: -len(INPUT_TOPIC_SUFFIX)]
    validate_job_name(job_name)
    return job_name


def derive_output_topic(job_name: str) -> str:
    validate_job_name(job_name)
    return f"{job_name}{OUTPUT_TOPIC_SUFFIX}"


def build_group_id(job_name: str) -> str:
    validate_job_name(job_name)
    return f"media-server-{job_name}"


def normalize_group_id(group_id: str | None, *, job_name: str) -> str:
    if group_id is None or not group_id.strip():
        return build_group_id(job_name)

    normalized = group_id.strip()
    if not GROUP_ID_PATTERN.match(normalized):
        raise ValueError("group_id contains invalid characters")
    return normalized
