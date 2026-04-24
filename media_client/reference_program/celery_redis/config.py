"""
config.py — 全局配置
"""
import os
from dataclasses import dataclass

@dataclass
class Settings:
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_RESULT_URL: str = os.getenv("REDIS_RESULT_URL", "redis://localhost:6379/1")

    # AES-256-GCM key (32字节，生产环境从 KMS 或 Vault 取)
    AES_KEY: bytes = bytes.fromhex(
        os.getenv("AES_KEY", "0" * 64)  # 64个hex字符 = 32字节
    )

    # 视频处理
    JPEG_QUALITY: int = 92          # 帧压缩质量（越高体积越大）
    FRAME_BATCH_SIZE: int = 1       # 每个 Celery task 处理几帧（通常1帧1task）
    OUTPUT_VIDEO_BITRATE: str = "8000k"  # 输出视频码率（从源视频探测后覆盖）
    TMP_DIR: str = "/tmp/video_pipeline"

settings = Settings()
