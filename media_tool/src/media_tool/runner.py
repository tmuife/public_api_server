import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from uuid import uuid4

from .errors import MediaToolError
from .image_io import ImageData, read_image, write_image
from .materials import plan_materials
from .video_io import probe_video, process_video

logger = logging.getLogger(__name__)


def peak_memory_mb() -> float:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(usage / (1024 * 1024 if sys.platform == "darwin" else 1024), 2)


def atomic_publish(source: Path, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".media-tool-", suffix=".part", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # 同一文件系统的硬链接以原子方式拒绝并发出现的已有目标。
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def dry_run(settings) -> list[dict]:
    plan = plan_materials(settings)
    result = []
    for item in plan:
        entry = {
            "input": str(item.source),
            "processed_output": str(item.processed),
            "unmatched_output": str(item.original),
            "kind": item.kind,
        }
        try:
            if item.kind == "image":
                read_image(item.source)
            else:
                probe_video(item.source)
        except MediaToolError as exc:
            entry["error"] = str(exc)
        result.append(entry)
    return result


def run_batch(settings, pipeline) -> tuple[int, Path]:
    plan = plan_materials(settings)
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    # mkdtemp 保证只清理确实由本次进程创建的目录。
    work = Path(tempfile.mkdtemp(prefix=run_id + "-", dir=settings.work_dir))
    report_dir = settings.output_material_path / ".media-tool-reports"
    if not report_dir.resolve().is_relative_to(settings.output_material_path):
        shutil.rmtree(work)
        raise MediaToolError("报告目录不能通过符号链接离开输出根目录")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (run_id + ".jsonl")
    failures = 0
    try:
        with report_path.open("x", encoding="utf-8") as report:
            for index, item in enumerate(plan):
                start = time.perf_counter()
                record = {
                    "input": str(item.source),
                    "detect_method": settings.detect_method,
                    "parameters": settings.report_dict(),
                }
                task_dir = work / str(index)
                try:
                    if not settings.overwrite and any(
                        p.exists() for p in {item.processed, item.original}
                    ):
                        record["status"] = "skipped_existing"
                        record["reason"] = "已有输出，未验证它来自本次配置"
                    else:
                        task_dir.mkdir()
                        if item.kind == "image":
                            image = read_image(item.source)
                            result = pipeline.process(image.frame)
                            record.update(
                                {
                                    "frames": 1,
                                    "detected": result.detected,
                                    "matched": result.matched,
                                    "swapped": result.swapped,
                                    "enhanced": result.enhanced,
                                    "matches": result.matches,
                                }
                            )
                            transformed = None
                            if result.swapped:
                                transformed = task_dir / item.processed.name
                                write_image(
                                    transformed,
                                    ImageData(result.frame, image.alpha),
                                    settings.image_jpeg_quality,
                                )
                        else:
                            transformed, counts = process_video(
                                item.source, task_dir, settings, pipeline
                            )
                            record.update(counts)
                        if transformed is not None:
                            target = item.processed
                            atomic_publish(transformed, target, settings.overwrite)
                            record.update(status="success", output=str(target))
                        elif settings.unmatched_action == "copy":
                            atomic_publish(item.source, item.original, settings.overwrite)
                            record.update(status="unmatched_copied", output=str(item.original))
                        else:
                            record["status"] = "unmatched_skipped"
                except Exception as exc:
                    failures += 1
                    record.update(status="failed", error=str(exc))
                    logger.error("素材处理失败 %s: %s", item.source, exc)
                except KeyboardInterrupt:
                    record.update(status="interrupted", error="用户中断")
                    raise
                finally:
                    record["seconds"] = round(time.perf_counter() - start, 4)
                    record["peak_memory_mb"] = peak_memory_mb()
                    record["fps"] = round(record.get("frames", 0) / max(record["seconds"], 1e-6), 3)
                    record["backends"] = pipeline.backend_info()
                    report.write(json.dumps(record, ensure_ascii=False) + "\n")
                    report.flush()
                    if settings.cleanup_work_dir and task_dir.exists():
                        shutil.rmtree(task_dir)
                    logger.info(
                        "%s %s（%.2fs）", record.get("status"), item.source, record["seconds"]
                    )
                if record["status"] == "failed" and settings.on_error == "stop":
                    break
    finally:
        if settings.cleanup_work_dir:
            shutil.rmtree(work)
    return (1 if failures else 0), report_path
