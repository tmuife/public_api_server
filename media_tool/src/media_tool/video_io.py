from dataclasses import dataclass
from fractions import Fraction
import itertools
import json
import logging
from pathlib import Path
import shutil
import subprocess
import time

import av
import cv2
import numpy as np

from .errors import MediaToolError

logger = logging.getLogger(__name__)


def run_process(arguments: list[str]) -> str:
    # stderr 落到临时文件，避免长视频日志无限占用内存；异常时也等待子进程退出。
    import tempfile

    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=errors)
        try:
            output, _ = process.communicate()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if process.returncode:
            errors.seek(max(0, errors.tell() - 4000))
            detail = errors.read().decode("utf-8", errors="replace")
            raise MediaToolError(f"{arguments[0]} 执行失败：{detail}")
        return output.decode("utf-8")


def check_tools() -> dict[str, str]:
    versions = {}
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            raise MediaToolError(f"缺少 {name}；macOS 请执行 brew install ffmpeg")
        versions[name] = run_process([name, "-version"]).splitlines()[0]
    return versions


@dataclass
class VideoInfo:
    video: dict
    audio: list[dict]
    rotation: int


def probe_video(path: Path) -> VideoInfo:
    data = json.loads(
        run_process(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)])
    )
    videos = [
        s
        for s in data["streams"]
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
    ]
    if not videos:
        raise MediaToolError(f"没有主视频流：{path}")
    video = videos[0]
    if video.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
        raise MediaToolError(f"首版不支持 HDR 视频：{path}")
    if not video.get("width") or not video.get("height"):
        raise MediaToolError(f"视频尺寸无效：{path}")
    rotation = float(video.get("tags", {}).get("rotate", 0))
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = float(side["rotation"])
    if abs(rotation - round(rotation / 90) * 90) > 0.01:
        raise MediaToolError("首版只支持 90 度倍数的视频旋转")
    return VideoInfo(
        video,
        [s for s in data["streams"] if s.get("codec_type") == "audio"],
        int(round(rotation)) % 360,
    )


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    # FFprobe display matrix 的正角度为逆时针。
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def frame_time(frame) -> Fraction:
    if frame.pts is None or frame.time_base is None:
        raise MediaToolError("视频帧缺少必要 PTS/time_base，不能确定时间线")
    return frame.pts * frame.time_base


def merge_audio(
    silent: Path, source: Path, destination: Path, info: VideoInfo, origin: Fraction
) -> None:
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-copyts", "-i", str(silent)]
    if info.audio:
        command += ["-itsoffset", f"{float(-origin):.12f}", "-i", str(source)]
    command += ["-map", "0:v:0", "-c:v", "copy"]
    for index, stream in enumerate(info.audio):
        command += ["-map", f"1:{stream['index']}"]
        codec = "copy" if stream["codec_name"] in {"aac", "mp3", "alac", "ac3", "eac3"} else "aac"
        command += [f"-c:a:{index}", codec]
        if codec == "aac":
            command += [f"-b:a:{index}", "192k"]
    command += [
        "-map_metadata",
        "-1",
        "-metadata:s:v:0",
        "rotate=0",
        "-avoid_negative_ts",
        "disabled",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    run_process(command)


def validate_video(
    source: Path,
    output: Path,
    info: VideoInfo,
    origin: Fraction,
    count: int,
    dimensions: tuple[int, int],
) -> None:
    final_info = probe_video(output)
    if len(final_info.audio) != len(info.audio):
        raise MediaToolError("输出音轨数量校验失败")
    if (final_info.video["width"], final_info.video["height"]) != dimensions or final_info.rotation:
        raise MediaToolError("输出展示尺寸或旋转信息校验失败")
    with av.open(str(source)) as original, av.open(str(output)) as converted:
        original_stream = original.streams[info.video["index"]]
        final_stream = converted.streams.video[0]
        tolerance = max(Fraction(1, 1000), 2 * final_stream.time_base)
        seen = 0
        for before, after in itertools.zip_longest(
            original.decode(original_stream), converted.decode(final_stream)
        ):
            if before is None or after is None:
                raise MediaToolError("输出展示帧数校验失败")
            if abs((frame_time(before) - origin) - frame_time(after)) > tolerance:
                raise MediaToolError(f"输出第 {seen} 帧时间戳偏离原时间线")
            seen += 1
        if seen != count:
            raise MediaToolError("输出展示帧数与处理帧数不一致")
    for before, after in zip(info.audio, final_info.audio):
        if "start_time" not in before or "start_time" not in after:
            raise MediaToolError("音轨缺少必要的起始时间，无法校验同步")
        offset = float(before["start_time"]) - float(origin)
        # AAC 编码延迟和封装取整允许最多 50 ms，但不允许任意音画移位。
        if abs(float(after["start_time"]) - offset) > 0.05:
            raise MediaToolError("输出音画起始偏移校验失败")
        if "duration" in before and "duration" in after:
            if abs(float(before["duration"]) - float(after["duration"])) > 0.1:
                raise MediaToolError("输出音轨时长校验失败")
    if "duration" in info.video and "duration" in final_info.video:
        if abs(float(info.video["duration"]) - float(final_info.video["duration"])) > 0.1:
            raise MediaToolError("输出视频结束时间校验失败")


def process_video(source: Path, work: Path, settings, pipeline) -> tuple[Path | None, dict]:
    info = probe_video(source)
    for audio in info.audio:
        if "start_time" not in audio:
            raise MediaToolError("音轨缺少 start_time，无法保留音画起始偏移")
    counts = {"frames": 0, "detected": 0, "matched": 0, "swapped": 0, "enhanced": 0}
    silent = work / "silent.mp4"
    with av.open(str(source)) as container:
        stream = container.streams[info.video["index"]]
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 2
        frames = container.decode(stream)
        first = next(frames, None)
        if first is None:
            raise MediaToolError("视频没有可解码帧")
        first_time = frame_time(first)
        audio_starts = [Fraction(a["start_time"]) for a in info.audio]
        origin = min([first_time, *audio_starts])
        time_base = stream.time_base
        rate = stream.average_rate or Fraction(30)
        width, height = first.width, first.height
        if info.rotation in {90, 270}:
            width, height = height, width
        padded_width, padded_height = width + width % 2, height + height % 2
        if (width, height) != (padded_width, padded_height):
            logger.info(
                "视频奇数尺寸补边：%s×%s -> %s×%s", width, height, padded_width, padded_height
            )
        dimensions = (padded_width, padded_height)
        with av.open(str(silent), "w", format="mp4") as output:
            encoded = output.add_stream(settings.video_encoder, rate=rate)
            encoded.width, encoded.height = dimensions
            encoded.pix_fmt = "yuv420p"
            encoded.time_base = encoded.codec_context.time_base = time_base
            sar = stream.sample_aspect_ratio or Fraction(1)
            encoded.codec_context.sample_aspect_ratio = (
                1 / sar if info.rotation in {90, 270} else sar
            )
            encoded.codec_context.thread_count = 2
            encoded.options = {
                "crf": str(settings.video_crf),
                "preset": settings.video_preset,
                "bf": "0",
            }
            encoded.codec_context.color_primaries = 1
            encoded.codec_context.color_trc = 1
            encoded.codec_context.colorspace = 1
            previous_time, pending = None, None
            last_progress = time.monotonic()

            def mux_packets(packets):
                nonlocal pending
                for packet in packets:
                    if pending is not None:
                        difference = packet.pts * packet.time_base - pending.pts * pending.time_base
                        pending.duration = max(1, round(difference / pending.time_base))
                        output.mux(pending)
                    pending = packet

            for frame in itertools.chain([first], frames):
                timestamp = frame_time(frame)
                if previous_time is not None and timestamp <= previous_time:
                    raise MediaToolError("视频展示时间戳必须严格递增")
                previous_time = timestamp
                pixels = frame.to_ndarray(format="bgr24")
                pixels = rotate_frame(pixels, info.rotation)
                if pixels.shape[:2] != (height, width):
                    raise MediaToolError("首版不支持视频中途变更分辨率")
                result = pipeline.process(pixels)
                for name in ("detected", "matched", "swapped", "enhanced"):
                    counts[name] += getattr(result, name)
                counts["frames"] += 1
                if time.monotonic() - last_progress >= 10:
                    logger.info(
                        "%s 已处理 %s 帧，成功换脸 %s 次",
                        source.name,
                        counts["frames"],
                        counts["swapped"],
                    )
                    last_progress = time.monotonic()
                if padded_width != width or padded_height != height:
                    result.frame = cv2.copyMakeBorder(
                        result.frame,
                        0,
                        padded_height - height,
                        0,
                        padded_width - width,
                        cv2.BORDER_REPLICATE,
                    )
                processed = av.VideoFrame.from_ndarray(result.frame, format="bgr24")
                processed = processed.reformat(format="yuv420p", dst_colorspace="ITU709")
                processed.pts = round((timestamp - origin) / time_base)
                processed.time_base = time_base
                mux_packets(encoded.encode(processed))
            mux_packets(encoded.encode())
            if pending is not None:
                end = (
                    (stream.start_time or 0) * time_base + stream.duration * time_base
                    if (stream.duration is not None)
                    else previous_time + 1 / rate
                )
                pending.duration = max(1, round((end - previous_time) / pending.time_base))
                output.mux(pending)
    if counts["swapped"] == 0:
        return None, counts
    final = work / "processed.mp4"
    merge_audio(silent, source, final, info, origin)
    validate_video(source, final, info, origin, counts["frames"], dimensions)
    return final, counts
