from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.config import AppSettings
from app.errors import ServiceError
from app.services.kafka_service import KafkaService
from app.utils.crypto import encrypt_bytes
from app.utils.ffmpeg import extract_frames_audio_metadata
from app.utils.job import derive_input_topic, generate_job_name
from app.utils.pathing import create_workspace

logger = logging.getLogger(__name__)


class VideoPreprocessService:
    def __init__(self, settings: AppSettings, kafka_service: KafkaService):
        self.settings = settings
        self.kafka_service = kafka_service

    def process_by_path(self, video_path: str) -> dict[str, Any]:
        source_path = Path(video_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise ServiceError(
                status_code=400,
                error_code="invalid_video_path",
                message="Video path does not exist or is not a file",
                details={"video_path": str(source_path)},
            )
        if not source_path.stat().st_size:
            raise ServiceError(
                status_code=400,
                error_code="empty_video_file",
                message="Video file is empty",
            )

        job_name = generate_job_name()
        workspace = create_workspace(self.settings.work_dir, job_name)

        normalized_source = workspace.source_dir / source_path.name
        shutil.copy2(source_path, normalized_source)

        return self._run_preprocess_pipeline(job_name=job_name, source_video=normalized_source)

    async def process_upload(self, upload_file: UploadFile) -> dict[str, Any]:
        if upload_file is None:
            raise ServiceError(
                status_code=400,
                error_code="missing_upload",
                message="Upload file is required",
            )

        if not upload_file.filename:
            raise ServiceError(
                status_code=400,
                error_code="invalid_upload",
                message="Upload filename is required",
            )

        if upload_file.content_type and not upload_file.content_type.lower().startswith("video/"):
            raise ServiceError(
                status_code=400,
                error_code="invalid_upload_content_type",
                message="Uploaded file must be a video payload",
                details={"content_type": upload_file.content_type},
            )

        job_name = generate_job_name()
        upload_path = await self._persist_upload(job_name=job_name, upload_file=upload_file)

        workspace = create_workspace(self.settings.work_dir, job_name)
        normalized_source = workspace.source_dir / upload_path.name
        shutil.copy2(upload_path, normalized_source)

        return self._run_preprocess_pipeline(job_name=job_name, source_video=normalized_source)

    async def _persist_upload(self, *, job_name: str, upload_file: UploadFile) -> Path:
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(upload_file.filename or "upload.mp4").suffix or ".mp4"
        filename = f"{job_name}{suffix}"
        destination = (self.settings.upload_dir / filename).resolve()

        with destination.open("wb") as buffer:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)

        await upload_file.close()

        if destination.stat().st_size <= 0:
            raise ServiceError(
                status_code=400,
                error_code="invalid_upload",
                message="Uploaded file is empty",
            )

        return destination

    def _run_preprocess_pipeline(self, *, job_name: str, source_video: Path) -> dict[str, Any]:
        workspace = create_workspace(self.settings.work_dir, job_name)
        input_topic = derive_input_topic(job_name)

        try:
            extraction = extract_frames_audio_metadata(
                source_path=source_video,
                frames_dir=workspace.frames_dir,
                audio_path=workspace.audio_dir / "source_audio.m4a",
                frame_format="jpg",
            )
        except Exception as exc:
            raise ServiceError(
                status_code=500,
                error_code="frame_extraction_failed",
                message="Failed to extract frames/audio from source video",
                details={"reason": str(exc)},
            ) from exc

        try:
            self.kafka_service.ensure_topic(input_topic)
        except ServiceError:
            raise

        messages: list[dict[str, Any]] = []
        for frame_index, frame_path in enumerate(extraction.frame_paths):
            encrypted = encrypt_bytes(frame_path.read_bytes(), self.settings.aes_key_bytes)
            messages.append(
                {
                    "job_name": job_name,
                    "frame_index": frame_index,
                    "nonce_b64": encrypted.nonce_b64,
                    "ciphertext_b64": encrypted.ciphertext_b64,
                    "tag_b64": encrypted.tag_b64,
                    "content_type": "image/jpeg",
                }
            )

        try:
            self.kafka_service.publish_frame_requests(topic=input_topic, messages=messages)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(
                status_code=500,
                error_code="frame_publish_failed",
                message="Failed to publish encrypted frame stream",
                details={"reason": str(exc)},
            ) from exc

        manifest = {
            "job_name": job_name,
            "input_topic": input_topic,
            "source_path": str(source_video.resolve()),
            "frame_count": len(extraction.frame_paths),
            "fps": extraction.metadata.fps,
            "bitrate": extraction.metadata.bitrate,
            "audio_path": str(extraction.audio_path.resolve()),
            "frame_format": extraction.metadata.frame_format,
        }

        workspace.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "job_name": job_name,
            "input_topic": input_topic,
            "manifest_summary": {
                "frame_count": manifest["frame_count"],
                "fps": manifest["fps"],
                "bitrate": manifest["bitrate"],
                "frame_format": manifest["frame_format"],
                "audio_path": manifest["audio_path"],
            },
        }
