from dataclasses import replace
import json

import numpy as np
from PIL import Image
import pytest

from media_tool.errors import MediaToolError
from media_tool.image_io import read_image, write_image
from media_tool.materials import plan_materials, scan_files
from media_tool.pipeline import FrameResult
from media_tool.runner import atomic_publish, run_batch


class NoMatch:
    def process(self, pixels):
        return FrameResult(pixels)

    def backend_info(self):
        return {"test": "CPU"}


def save_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 16), "red").save(path)


def test_recursive_scan_ignores_nonmedia_and_links(settings):
    image = settings.input_material_path / "中文/嵌套/a.png"
    save_image(image)
    (settings.input_material_path / "note.txt").write_text("note")
    (settings.input_material_path / "linked.png").symlink_to(image)
    assert scan_files(settings.input_material_path, {".png"}) == [image]
    assert (
        plan_materials(settings)[0].processed == settings.output_material_path / "中文/嵌套/a.png"
    )


def test_output_collision_and_casefold(settings):
    (settings.input_material_path / "a.mov").touch()
    (settings.input_material_path / "a.mov.mp4").touch()
    with pytest.raises(MediaToolError, match="冲突"):
        plan_materials(settings)
    (settings.input_material_path / "a.mov.mp4").unlink()
    (settings.input_material_path / "A.MOV").touch()
    with pytest.raises(MediaToolError, match="冲突"):
        plan_materials(settings)


def test_output_symlink_cannot_escape(settings, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (settings.output_material_path / "sub").symlink_to(outside, target_is_directory=True)
    save_image(settings.input_material_path / "sub/a.png")
    with pytest.raises(MediaToolError, match="离开输出"):
        plan_materials(settings)


def test_alpha_and_exif_orientation(tmp_path):
    path = tmp_path / "alpha.png"
    values = np.zeros((12, 10, 4), np.uint8)
    values[:, :, 0] = 255
    values[:, :, 3] = np.arange(10) * 20
    Image.fromarray(values).save(path)
    data = read_image(path)
    destination = tmp_path / "result.png"
    write_image(destination, data, 95)
    assert np.array_equal(np.array(Image.open(destination))[:, :, 3], values[:, :, 3])
    image = Image.new("RGB", (10, 20))
    exif = Image.Exif()
    exif[274] = 6
    oriented = tmp_path / "oriented.jpg"
    image.save(oriented, exif=exif)
    assert read_image(oriented).frame.shape == (10, 20, 3)


def test_reject_animated_and_multipage(tmp_path):
    path = tmp_path / "animated.webp"
    Image.new("RGB", (10, 10), "red").save(
        path, save_all=True, append_images=[Image.new("RGB", (10, 10), "blue")], duration=100
    )
    with pytest.raises(MediaToolError, match="动画"):
        read_image(path)


def test_static_heic_and_multipage_tiff(tmp_path):
    heic = tmp_path / "sample.heic"
    Image.new("RGB", (30, 20), "green").save(heic, format="HEIF")
    assert read_image(heic).frame.shape == (20, 30, 3)
    tiff = tmp_path / "pages.tiff"
    Image.new("RGB", (10, 10), "red").save(
        tiff, save_all=True, append_images=[Image.new("RGB", (10, 10), "blue")]
    )
    with pytest.raises(MediaToolError, match="多页"):
        read_image(tiff)


def test_atomic_no_overwrite_preserves_original(tmp_path):
    source, destination = tmp_path / "new", tmp_path / "old"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        atomic_publish(source, destination, False)
    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob(".media-tool-*"))
    atomic_publish(source, destination, True)
    assert destination.read_bytes() == b"new"


@pytest.mark.parametrize("action", ["copy", "skip"])
def test_unmatched_policy_and_safe_cleanup(settings, action):
    source = settings.input_material_path / "子目录/a.png"
    save_image(source)
    user_file = settings.work_dir / "用户文件.txt"
    user_file.write_text("keep")
    code, report = run_batch(replace(settings, unmatched_action=action), NoMatch())
    assert code == 0
    record = json.loads(report.read_text())
    target = settings.output_material_path / "子目录/a.png"
    if action == "copy":
        assert target.read_bytes() == source.read_bytes()
    else:
        assert not target.exists()
    assert record["status"] == ("unmatched_copied" if action == "copy" else "unmatched_skipped")
    assert list(settings.work_dir.iterdir()) == [user_file]
    assert report.exists()


@pytest.mark.parametrize("stop,expected", [(False, 3), (True, 2)])
def test_continue_or_stop_failure(settings, stop, expected):
    save_image(settings.input_material_path / "1.png")
    (settings.input_material_path / "2.png").write_bytes(b"corrupted")
    save_image(settings.input_material_path / "3.png")
    code, report = run_batch(replace(settings, on_error="stop" if stop else "continue"), NoMatch())
    records = [json.loads(line) for line in report.read_text().splitlines()]
    assert code == 1 and len(records) == expected
    assert records[1]["status"] == "failed"


def test_interruption_records_and_cleans(settings):
    save_image(settings.input_material_path / "a.png")

    class Interrupted(NoMatch):
        def process(self, frame):
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run_batch(settings, Interrupted())
    assert not list(settings.work_dir.iterdir())
    report = next((settings.output_material_path / ".media-tool-reports").glob("*.jsonl"))
    assert json.loads(report.read_text())["status"] == "interrupted"
    assert not (settings.output_material_path / "a.png").exists()
