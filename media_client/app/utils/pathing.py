from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobWorkspace:
    job_name: str
    job_dir: Path
    source_dir: Path
    frames_dir: Path
    audio_dir: Path
    output_dir: Path
    meta_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.meta_dir / "manifest.json"


def _build_workspace(work_dir: Path, job_name: str) -> JobWorkspace:
    jobs_root = work_dir / "jobs"
    job_dir = jobs_root / job_name

    return JobWorkspace(
        job_name=job_name,
        job_dir=job_dir,
        source_dir=job_dir / "source",
        frames_dir=job_dir / "frames",
        audio_dir=job_dir / "audio",
        output_dir=job_dir / "output",
        meta_dir=job_dir / "meta",
    )


def ensure_directory_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    probe = path / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Directory is not writable: {path}") from exc
    finally:
        if probe.exists():
            probe.unlink()


def create_workspace(work_dir: Path, job_name: str) -> JobWorkspace:
    workspace = _build_workspace(work_dir, job_name)

    for directory in (
        workspace.source_dir,
        workspace.frames_dir,
        workspace.audio_dir,
        workspace.output_dir,
        workspace.meta_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return workspace


def resolve_workspace(work_dir: Path, job_name: str) -> JobWorkspace:
    workspace = _build_workspace(work_dir, job_name)
    if not workspace.job_dir.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace.job_dir}")
    return workspace
