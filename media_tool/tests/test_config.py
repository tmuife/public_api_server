from dataclasses import replace

import pytest

from media_tool.config import load_settings
from media_tool.errors import MediaToolError


def test_env_paths_resolve_against_file(env_file, settings, tmp_path, monkeypatch):
    env_file.write_text(env_file.read_text().replace(str(settings.input_material_path), "./input"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("detect_method", "yunet")
    loaded = load_settings(env_file)
    assert loaded.input_material_path == settings.input_material_path
    assert loaded.detect_method == "yunet"
    assert load_settings(env_file, {"detect_method": "insightface"}).detect_method == "insightface"


@pytest.mark.parametrize(
    "values",
    [
        {"thresholds": float("nan")},
        {"thresholds": 0},
        {"enhance_blend": 1.1},
        {"detect_method": "insight"},
        {"detection_max_side": 63},
        {"overwrite": False, "video_crf": 52},
    ],
)
def test_reject_invalid_config(settings, values):
    with pytest.raises(MediaToolError):
        replace(settings, **values).validate()


def test_reject_nested_and_symlink_directory(settings):
    nested = replace(settings, output_material_path=settings.input_material_path / "output")
    with pytest.raises(MediaToolError, match="互相包含"):
        nested.validate()
    linked = settings.work_dir / "linked"
    linked.symlink_to(settings.input_material_path, target_is_directory=True)
    with pytest.raises(MediaToolError):
        replace(settings, output_material_path=linked.resolve()).validate()


def test_only_selected_models_required(settings):
    for name in [
        "detection/yunet/face_detection_yunet_2023mar.onnx",
        "recognition/w600k_r50.onnx",
        "swap/inswapper_128.onnx",
    ]:
        path = settings.models_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert len(replace(settings, detect_method="yunet").required_models()) == 3
    with pytest.raises(MediaToolError, match="det_10g"):
        settings.required_models()


def test_reject_unknown_and_malformed_env(env_file):
    env_file.write_text(env_file.read_text() + "\noverwirte=true\n")
    with pytest.raises(MediaToolError, match="未知配置项"):
        load_settings(env_file)
