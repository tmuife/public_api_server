"""
api.py — FastAPI 接口

启动：
  uvicorn api:app --reload

Worker 启动（另一个终端）：
  celery -A celery_app worker --loglevel=info --concurrency=8
"""
import uuid
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from celery.result import AsyncResult, GroupResult

from celery_app import app as celery_app
from tasks import produce_video
from config import settings

log = logging.getLogger(__name__)
api = FastAPI(title="Video Pipeline")


# ─────────────────────────────────────────────
# 请求/响应模型
# ─────────────────────────────────────────────
class SubmitResponse(BaseModel):
    video_id: str
    job_id: str
    message: str


class ProgressResponse(BaseModel):
    video_id: str
    job_id: str
    state: Literal["PENDING", "STARTED", "SUCCESS", "FAILURE", "REVOKED"]
    progress_pct: float          # 0.0 ~ 100.0
    completed_frames: int
    total_frames: int
    output_path: str | None      # 完成后才有值
    error: str | None


# ─────────────────────────────────────────────
# POST /videos/submit
# ─────────────────────────────────────────────
@api.post("/videos/submit", response_model=SubmitResponse)
async def submit_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    上传视频，开始处理流程。
    返回 job_id 用于后续进度查询。
    """
    video_id = str(uuid.uuid4())
    save_dir = Path(settings.TMP_DIR) / video_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存上传文件
    video_path = save_dir / file.filename
    video_path.write_bytes(await file.read())

    # 在后台线程启动（produce_video 会阻塞读帧，不能在 async 里直接跑）
    background_tasks.add_task(_start_pipeline, str(video_path), video_id)

    return SubmitResponse(
        video_id=video_id,
        job_id="pending",   # 真正的 job_id 在后台产生，客户端可轮询 /videos/{video_id}/progress
        message="Processing started",
    )


def _start_pipeline(video_path: str, video_id: str):
    """后台线程：调用 produce_video，存储 job_id"""
    try:
        job_id = produce_video(video_path, video_id)
        # 存 job_id 到 Redis，供进度查询接口使用
        import redis
        r = redis.from_url(settings.REDIS_URL)
        r.set(f"pipeline:job_id:{video_id}", job_id, ex=86400)
        log.info(f"[{video_id}] pipeline started, job_id={job_id}")
    except Exception as e:
        log.error(f"[{video_id}] pipeline failed to start: {e}")


# ─────────────────────────────────────────────
# GET /videos/{video_id}/progress
# ─────────────────────────────────────────────
@api.get("/videos/{video_id}/progress", response_model=ProgressResponse)
def get_progress(video_id: str):
    """
    查询处理进度。
    Celery chord 完成后 state = SUCCESS，output_path 有值。
    """
    import redis
    r = redis.from_url(settings.REDIS_URL)
    job_id_bytes = r.get(f"pipeline:job_id:{video_id}")

    if not job_id_bytes:
        raise HTTPException(404, "video_id not found or pipeline not started yet")

    job_id = job_id_bytes.decode()

    # chord 的 AsyncResult（指向 assemble_video 这个 callback task）
    result = AsyncResult(job_id, app=celery_app)

    # 尝试获取 Group 进度（chord header 的进度）
    completed = 0
    total = 0
    try:
        # chord 的 parent 是 GroupResult（所有 process_frame 的集合）
        group_result: GroupResult = result.parent
        if group_result:
            total = len(group_result.results)
            completed = sum(1 for r in group_result.results if r.ready())
    except Exception:
        pass

    progress_pct = (completed / total * 100) if total > 0 else 0.0

    return ProgressResponse(
        video_id=video_id,
        job_id=job_id,
        state=result.state,
        progress_pct=round(progress_pct, 1),
        completed_frames=completed,
        total_frames=total,
        output_path=result.result if result.state == "SUCCESS" else None,
        error=str(result.result) if result.state == "FAILURE" else None,
    )


# ─────────────────────────────────────────────
# GET /videos/{video_id}/download
# ─────────────────────────────────────────────
@api.get("/videos/{video_id}/download")
def download_video(video_id: str):
    """处理完成后下载视频"""
    output_path = Path(settings.TMP_DIR) / video_id / "output.mp4"
    if not output_path.exists():
        raise HTTPException(404, "Output not ready yet")
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"{video_id}_processed.mp4",
    )
