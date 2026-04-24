"""
tasks.py — 所有 Celery Task 定义

Task 调用链：
  produce_video()
    └─ chord(process_frame.s() × N)(assemble_video.s())
                                          ↑
                          所有帧处理完后自动触发，results 是有序帧列表
"""
import io
import os
import json
import subprocess
import logging
from pathlib import Path

from celery import shared_task, chord, group
from PIL import Image

from celery_app import app
from crypto import encrypt, decrypt
from config import settings

log = logging.getLogger(__name__)
Path(settings.TMP_DIR).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 工具：探测视频信息
# ─────────────────────────────────────────────
def probe_video(video_path: str) -> dict:
    """用 ffprobe 获取视频流信息，返回 dict"""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")

    # 解析帧率（格式为 "30/1" 或 "2997/100"）
    num, den = map(int, video["r_frame_rate"].split("/"))
    fps = num / den

    # 探测码率（如果视频流没有就用容器码率）
    bitrate = video.get("bit_rate") or "8000000"

    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "bitrate": f"{int(int(bitrate) / 1000)}k",  # 转换为 "8000k" 格式
        "codec": video["codec_name"],
    }


def count_frames(video_path: str) -> int:
    """精确统计视频总帧数（走 ffprobe 计包数，比解码快）"""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


# ─────────────────────────────────────────────
# Task 1：单帧处理（decrypt → 图像处理 → encrypt）
# ─────────────────────────────────────────────
@app.task(bind=True, max_retries=3, default_retry_delay=5)
def process_frame(self, encrypted_frame: bytes, frame_index: int, video_id: str) -> dict:
    """
    单帧处理 Task。
    返回: {"frame_index": int, "data": bytes(encrypted), "video_id": str}
    bind=True 允许 self.retry()
    """
    try:
        # 1. 解密
        jpeg_bytes = decrypt(encrypted_frame)

        # 2. 图像处理（在这里实现你的业务逻辑）
        processed_jpeg = _process_image(jpeg_bytes)

        # 3. 加密结果
        encrypted_result = encrypt(processed_jpeg)

        return {
            "frame_index": frame_index,
            "data": encrypted_result,
            "video_id": video_id,
        }

    except Exception as exc:
        log.warning(f"[{video_id}] frame {frame_index} failed: {exc}, retrying...")
        raise self.retry(exc=exc)


def _process_image(jpeg_bytes: bytes) -> bytes:
    """
    ★ 在这里实现你的图像处理逻辑 ★
    输入/输出都是 JPEG bytes。
    示例：转灰度（替换成你的 AI 推理、打码、滤镜等）
    """
    img = Image.open(io.BytesIO(jpeg_bytes))

    # ---- 你的处理逻辑 ----
    # img = img.convert("L").convert("RGB")  # 转灰度示例
    # img = your_ai_model.infer(img)
    # ----------------------

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=settings.JPEG_QUALITY)
    return buf.getvalue()


# ─────────────────────────────────────────────
# Task 2：汇总 + 合并（chord 回调，所有帧完成后自动触发）
# ─────────────────────────────────────────────
@app.task
def assemble_video(results: list[dict], video_id: str, meta: dict) -> str:
    """
    chord 回调。results 是所有 process_frame 的返回值列表（无序）。
    meta: {"fps": float, "width": int, "height": int, "bitrate": str, "audio_path": str}
    返回: 最终输出视频路径
    """
    log.info(f"[{video_id}] assembling {len(results)} frames...")

    frames_dir = Path(settings.TMP_DIR) / video_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 1. 按 frame_index 排序并解密写盘
    for item in sorted(results, key=lambda x: x["frame_index"]):
        jpeg = decrypt(item["data"])
        frame_path = frames_dir / f"frame_{item['frame_index']:08d}.jpg"
        frame_path.write_bytes(jpeg)

    # 2. ffmpeg：帧序列 → 无声视频（保持原码率）
    tmp_video = Path(settings.TMP_DIR) / video_id / "video_only.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(meta["fps"]),
            "-i", str(frames_dir / "frame_%08d.jpg"),
            "-c:v", "libx264",
            "-b:v", meta["bitrate"],      # 原始码率
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(tmp_video),
        ],
        check=True,
        capture_output=True,
    )

    # 3. ffmpeg：合并音频
    output_path = Path(settings.TMP_DIR) / video_id / "output.mp4"
    audio_path = meta.get("audio_path", "")

    if audio_path and Path(audio_path).exists():
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(tmp_video),
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "copy",
                "-shortest",             # 以最短流为准（防止音视频长度微小差异）
                str(output_path),
            ],
            check=True, capture_output=True,
        )
    else:
        # 没有音频直接用无声视频
        output_path = tmp_video

    log.info(f"[{video_id}] done → {output_path}")
    return str(output_path)


# ─────────────────────────────────────────────
# 入口：生产者（拆帧 + 派发 chord）
# ─────────────────────────────────────────────
def produce_video(video_path: str, video_id: str) -> str:
    """
    拆分视频，为每帧创建一个 Celery task，用 chord 保证全部完成后自动汇总。
    返回: chord 的 AsyncResult id（用于进度查询）
    """
    video_path = str(Path(video_path).resolve())

    # ── 1. 探测视频信息 ──────────────────────
    meta = probe_video(video_path)
    log.info(f"[{video_id}] {meta}")

    # ── 2. 分离音频 ──────────────────────────
    audio_path = str(Path(settings.TMP_DIR) / video_id / "audio.aac")
    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",                   # 只要音频
            "-acodec", "copy",       # 不重编码
            audio_path,
        ],
        check=True, capture_output=True,
    )
    meta["audio_path"] = audio_path

    # ── 3. 拆帧：逐帧读取 raw RGB → JPEG压缩 → AES加密 ──
    w, h = meta["width"], meta["height"]
    frame_size = w * h * 3          # raw RGB 每帧字节数

    cmd = [
        "ffmpeg", "-i", video_path,
        "-an",                       # 去音频
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    tasks = []
    frame_index = 0

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        # JPEG 压缩（大幅降低 Redis 存储体积）
        img = Image.frombytes("RGB", (w, h), raw)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=settings.JPEG_QUALITY)
        jpeg_bytes = buf.getvalue()

        # AES 加密
        encrypted = encrypt(jpeg_bytes)

        # 创建 Celery task signature（不立即执行）
        tasks.append(
            process_frame.s(encrypted, frame_index, video_id)
        )
        frame_index += 1

    proc.wait()
    log.info(f"[{video_id}] {frame_index} frames dispatched")

    # ── 4. chord：所有帧处理完后自动触发 assemble_video ──
    #    assemble_video 会收到 results 列表（所有 process_frame 返回值）
    job = chord(
        group(tasks)
    )(
        assemble_video.s(video_id=video_id, meta=meta)
    )

    return job.id   # 用这个 id 轮询进度
