from dataclasses import replace
from fractions import Fraction
import subprocess

import av
import numpy as np
import pytest

from media_tool.errors import MediaToolError
from media_tool.pipeline import FrameResult
from media_tool.video_io import frame_time, probe_video, process_video, rotate_frame


class Matched:
    def process(self, frame):
        return FrameResult(frame, 1, 1, 1, 0)


def create_video(path, *, audio="none", offset=0, vfr=False, hdr=False, video_offset=0):
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if video_offset:
        command += ["-itsoffset", str(video_offset)]
    command += ["-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=1"]
    tracks = 2 if audio == "multiple" else (0 if audio == "none" else 1)
    for index in range(tracks):
        if offset:
            command += ["-itsoffset", str(offset)]
        command += ["-f", "lavfi", "-i", f"sine=frequency={440 + 220 * index}:duration=1"]
    command += ["-map", "0:v:0"]
    for index in range(tracks):
        command += ["-map", f"{index + 1}:a:0"]
    if vfr:
        command += ["-vf", "select='not(mod(n,3))+eq(n,1)'", "-fps_mode", "vfr"]
    elif video_offset:
        command += ["-fps_mode", "vfr"]
    command += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if hdr:
        command += ["-color_trc", "smpte2084"]
    if tracks:
        command += ["-c:a", "pcm_s16le" if audio == "pcm" else "aac"]
    command.append(str(path))
    subprocess.run(command, check=True, capture_output=True)


@pytest.mark.parametrize(
    "audio,offset,vfr,video_offset",
    [
        ("none", 0, False, 0),
        ("aac", 0, False, 0),
        ("aac", 0.5, False, 0),
        ("multiple", 0, False, 0),
        ("pcm", 0, False, 0),
        ("none", 0, True, 0),
        ("aac", 0.5, True, 0),
        ("aac", 0, False, 0.5),
    ],
)
def test_timeline_and_audio(settings, tmp_path, audio, offset, vfr, video_offset):
    source = tmp_path / ("source.mkv" if audio == "pcm" else "source.mp4")
    create_video(source, audio=audio, offset=offset, vfr=vfr, video_offset=video_offset)
    work = tmp_path / "job"
    work.mkdir()
    output, counts = process_video(source, work, settings, Matched())
    info = probe_video(output)
    assert len(info.audio) == (2 if audio == "multiple" else (0 if audio == "none" else 1))
    assert counts["frames"] == (5 if vfr else 10)
    if vfr:
        with av.open(str(output)) as container:
            times = [frame_time(f) for f in container.decode(video=0)]
        intervals = [b - a for a, b in zip(times, times[1:])]
        assert len(set(intervals)) > 1
    if offset:
        assert (
            abs(
                float(info.audio[0]["start_time"])
                - float(probe_video(source).audio[0]["start_time"])
            )
            < 0.05
        )
    if video_offset:
        assert float(info.video["start_time"]) >= 0.45
        assert float(info.audio[0]["start_time"]) < 0.05
    if audio == "pcm":
        assert info.audio[0]["codec_name"] == "aac"


def test_rotation_and_odd_dimension(settings, tmp_path):
    source = tmp_path / "source.mp4"
    create_video(source)
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(source),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )
    work = tmp_path / "job"
    work.mkdir()
    output, _ = process_video(rotated, work, settings, Matched())
    result = probe_video(output)
    assert (result.video["width"], result.video["height"]) == (48, 64)
    assert result.rotation == 0
    pixels = np.arange(2 * 3 * 3).reshape(2, 3, 3)
    assert np.array_equal(rotate_frame(pixels, 90), np.rot90(pixels))


def test_unmatched_checks_final_frame(settings, tmp_path):
    source = tmp_path / "source.mp4"
    create_video(source)

    class FinalMatch:
        count = 0

        def process(self, frame):
            self.count += 1
            return FrameResult(frame, 1, int(self.count == 10), int(self.count == 10), 0)

    output, counts = process_video(source, tmp_path, settings, FinalMatch())
    assert output is not None and counts["frames"] == 10 and counts["swapped"] == 1


def test_hdr_and_missing_timestamps(settings, tmp_path):
    source = tmp_path / "hdr.mp4"
    create_video(source, hdr=True)
    with pytest.raises(MediaToolError, match="HDR"):
        probe_video(source)
    frame = av.VideoFrame(32, 32)
    frame.time_base = Fraction(1, 30)
    with pytest.raises(MediaToolError, match="PTS"):
        frame_time(frame)


def test_odd_dimension_padding(settings, tmp_path):
    source = tmp_path / "odd.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=65x49:rate=2:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv444p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    output, _ = process_video(source, tmp_path, settings, Matched())
    info = probe_video(output)
    assert (info.video["width"], info.video["height"]) == (66, 50)


def test_merge_failure_does_not_publish(settings, tmp_path, monkeypatch):
    from media_tool.runner import run_batch
    import media_tool.video_io as video_io

    create_video(settings.input_material_path / "source.mp4")

    def fail(*args):
        raise MediaToolError("合并失败")

    monkeypatch.setattr(video_io, "merge_audio", fail)
    pipeline = Matched()
    pipeline.backend_info = lambda: {}
    code, report = run_batch(settings, pipeline)
    assert code == 1 and "合并失败" in report.read_text()
    assert not (settings.output_material_path / "source.mp4").exists()


def test_unmatched_video_is_byte_copy(settings, tmp_path):
    from media_tool.runner import run_batch

    source = settings.input_material_path / "source.mp4"
    create_video(source, audio="aac")

    class NoMatch:
        def process(self, frame):
            return FrameResult(frame)

        def backend_info(self):
            return {}

    code, _ = run_batch(replace(settings, unmatched_action="copy"), NoMatch())
    assert code == 0
    assert (settings.output_material_path / "source.mp4").read_bytes() == source.read_bytes()
