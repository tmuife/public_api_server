"""
celery_app.py — Celery 实例初始化
"""
from celery import Celery
from config import settings

app = Celery(
    "video_pipeline",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_RESULT_URL,
)

app.conf.update(
    # 序列化（传大量 bytes 用 pickle 更高效）
    task_serializer="pickle",
    result_serializer="pickle",
    accept_content=["pickle", "json"],

    # 超时设置
    task_soft_time_limit=300,    # 单帧处理超过5分钟发软超时信号
    task_time_limit=360,         # 再过60秒强制终止

    # Worker 并发（每个进程独立，不共享内存，稳定）
    worker_prefetch_multiplier=1,  # 不预取，处理完一个再拿下一个（适合重任务）

    # 结果保留时间（assembler 取完就可以丢）
    result_expires=3600,  # 1小时

    # 失败重试
    task_acks_late=True,           # 任务执行完才 ack，崩溃会重试
    task_reject_on_worker_lost=True,
)
