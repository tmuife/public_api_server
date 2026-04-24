from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    bitrate: int
    width: int
    height: int
    frame_count: int
    frame_format: str


@dataclass(frozen=True)
class ExtractionResult:
    frame_paths: list[Path]
    audio_path: Path
    metadata: VideoMetadata


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(stderr or "command execution failed")


def _parse_fps(raw: str) -> float:
    if not raw:
        return 24.0
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        try:
            numerator_value = float(numerator)
            denominator_value = float(denominator)
            if denominator_value == 0:
                return 24.0
            return numerator_value / denominator_value
        except ValueError:
            return 24.0
    try:
        return float(raw)
    except ValueError:
        return 24.0


def probe_video(source_path: Path, *, frame_count_fallback: int, frame_format: str) -> VideoMetadata:
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,bit_rate,width,height,nb_frames",
        "-of",
        "json",
        str(source_path),
    ]

    completed = subprocess.run(probe_cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(stderr or "ffprobe failed")

    try:
        payload = json.loads(completed.stdout or "{}")
    except ValueError as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc

    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe returned no video stream")

    stream = streams[0]
    fps = _parse_fps(str(stream.get("avg_frame_rate") or ""))

    bitrate_raw = stream.get("bit_rate")
    try:
        bitrate = int(bitrate_raw) if bitrate_raw is not None else 1_000_000
    except ValueError:
        bitrate = 1_000_000

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)

    nb_frames = stream.get("nb_frames")
    try:
        frame_count = int(nb_frames) if nb_frames is not None else frame_count_fallback
    except ValueError:
        frame_count = frame_count_fallback

    if frame_count <= 0:
        frame_count = frame_count_fallback

    return VideoMetadata(
        fps=fps,
        bitrate=max(1, bitrate),
        width=width,
        height=height,
        frame_count=frame_count,
        frame_format=frame_format,
    )


def extract_frames_audio_metadata(
    source_path: Path,
    *,
    frames_dir: Path,
    audio_path: Path,
    frame_format: str = "jpg",
) -> ExtractionResult:
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    frame_pattern = frames_dir / f"frame_%06d.{frame_format}"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-vsync",
            "0",
            str(frame_pattern),
        ]
    )

    frame_paths = sorted(frames_dir.glob(f"frame_*.{frame_format}"))
    if not frame_paths:
        raise RuntimeError("No frames extracted from source video")

    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-c:a",
            "aac",
            str(audio_path),
        ]
    )

    metadata = probe_video(
        source_path,
        frame_count_fallback=len(frame_paths),
        frame_format=frame_format,
    )

    return ExtractionResult(
        frame_paths=frame_paths,
        audio_path=audio_path,
        metadata=metadata,
    )


def compose_video(
    *,
    frames_dir: Path,
    frame_format: str,
    fps: float,
    bitrate: int,
    audio_path: Path,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_pattern = frames_dir / f"frame_%06d.{frame_format}"
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_pattern),
    ]

    if audio_path.exists():
        command.extend(["-i", str(audio_path), "-c:a", "aac"])

    command.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            str(int(bitrate)),
            str(output_path),
        ]
    )

    _run(command)
    return output_path.resolve()
