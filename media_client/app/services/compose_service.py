from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import AppSettings
from app.errors import ServiceError
from app.services.kafka_service import KafkaService
from app.utils.crypto import decrypt_bytes
from app.utils.ffmpeg import compose_video
from app.utils.job import derive_job_name_from_topic, validate_topic_name
from app.utils.pathing import resolve_workspace


class VideoComposeService:
    def __init__(self, settings: AppSettings, kafka_service: KafkaService):
        self.settings = settings
        self.kafka_service = kafka_service

    def compose(self, topic: str) -> dict[str, Any]:
        normalized_topic = topic.strip()
        if not normalized_topic:
            raise ServiceError(
                status_code=400,
                error_code="invalid_topic",
                message="topic must not be empty",
            )

        try:
            validate_topic_name(normalized_topic)
        except ValueError as exc:
            raise ServiceError(
                status_code=400,
                error_code="invalid_topic",
                message=str(exc),
            ) from exc

        job_name = derive_job_name_from_topic(normalized_topic)
        workspace = self._resolve_workspace_or_404(job_name)
        manifest = self._load_manifest_or_404(workspace.manifest_path)

        if manifest.get("job_name") != job_name:
            raise ServiceError(
                status_code=409,
                error_code="manifest_job_mismatch",
                message="Manifest job_name does not match compose topic",
                details={
                    "manifest_job_name": manifest.get("job_name"),
                    "topic": normalized_topic,
                },
            )

        expected_count = int(manifest.get("frame_count") or 0)
        if expected_count <= 0:
            raise ServiceError(
                status_code=409,
                error_code="invalid_manifest",
                message="Manifest frame_count must be greater than 0",
            )

        try:
            frames_by_index = self.kafka_service.consume_frame_results(
                topic=normalized_topic,
                job_name=job_name,
                expected_count=expected_count,
                timeout_seconds=self.settings.kafka.consume_timeout_seconds,
            )
        except ServiceError:
            raise

        output_frames_dir = workspace.output_dir / "compose_frames"
        output_frames_dir.mkdir(parents=True, exist_ok=True)

        frame_format = str(manifest.get("frame_format") or "jpg")

        for frame_index in range(expected_count):
            payload = frames_by_index.get(frame_index)
            if payload is None:
                raise ServiceError(
                    status_code=409,
                    error_code="missing_frame",
                    message="Frame sequence has missing indexes",
                    details={"frame_index": frame_index},
                )

            try:
                plain_frame = decrypt_bytes(
                    nonce_b64=payload["nonce_b64"],
                    ciphertext_b64=payload["ciphertext_b64"],
                    tag_b64=payload["tag_b64"],
                    aes_key=self.settings.aes_key_bytes,
                )
            except Exception as exc:
                raise ServiceError(
                    status_code=409,
                    error_code="frame_decrypt_failed",
                    message="Failed to decrypt frame payload",
                    details={"frame_index": frame_index, "reason": str(exc)},
                ) from exc

            output_frame_path = output_frames_dir / f"frame_{frame_index + 1:06d}.{frame_format}"
            output_frame_path.write_bytes(plain_frame)

        audio_path = Path(str(manifest.get("audio_path") or "")).expanduser().resolve()
        if not audio_path.exists():
            raise ServiceError(
                status_code=404,
                error_code="audio_not_found",
                message="Source audio file referenced by manifest does not exist",
                details={"audio_path": str(audio_path)},
            )

        fps = float(manifest.get("fps") or 24.0)
        bitrate = int(manifest.get("bitrate") or 1_000_000)
        output_path = workspace.output_dir / "final.mp4"

        try:
            composed_path = compose_video(
                frames_dir=output_frames_dir,
                frame_format=frame_format,
                fps=fps,
                bitrate=bitrate,
                audio_path=audio_path,
                output_path=output_path,
            )
        except Exception as exc:
            raise ServiceError(
                status_code=500,
                error_code="video_compose_failed",
                message="Failed to compose final video",
                details={"reason": str(exc)},
            ) from exc

        return {
            "job_name": job_name,
            "topic": normalized_topic,
            "output_path": str(composed_path),
        }

    def _resolve_workspace_or_404(self, job_name: str):
        try:
            workspace = resolve_workspace(self.settings.work_dir, job_name)
        except FileNotFoundError as exc:
            raise ServiceError(
                status_code=404,
                error_code="job_not_found",
                message="No workspace found for topic",
                details={"job_name": job_name},
            ) from exc
        return workspace

    @staticmethod
    def _load_manifest_or_404(manifest_path: Path) -> dict[str, Any]:
        if not manifest_path.exists():
            raise ServiceError(
                status_code=404,
                error_code="manifest_not_found",
                message="Manifest file not found for job",
                details={"manifest_path": str(manifest_path)},
            )

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ServiceError(
                status_code=409,
                error_code="invalid_manifest",
                message="Manifest is not valid JSON",
                details={"manifest_path": str(manifest_path)},
            ) from exc

        return payload
