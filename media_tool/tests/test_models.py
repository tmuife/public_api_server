"""可选真实模型验收；普通测试不下载模型或图片。"""

import os
from pathlib import Path

import numpy as np
import pytest

from media_tool.config import Settings
from media_tool.image_io import read_image
from media_tool.face_analyzer import FaceAnalyzer
from media_tool.face_enhancer import FaceEnhancer
from media_tool.face_swapper import FaceSwapper

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        os.environ.get("MEDIA_TOOL_MODEL_TESTS") != "1", reason="需要显式启用真实模型验证"
    ),
]


@pytest.mark.parametrize("method", ["insightface", "yunet"])
def test_real_detection_recognition_swap(method, tmp_path):
    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        tmp_path / "input",
        tmp_path / "work",
        tmp_path / "output",
        tmp_path / "reference",
        tmp_path / "target",
        root / "models",
        detect_method=method,
        onnx_provider="cpu",
        enhance_enabled=False,
    )
    source = read_image(Path(os.environ["MEDIA_TOOL_TEST_FACE"])).frame
    target = read_image(Path(os.environ["MEDIA_TOOL_TEST_TARGET"])).frame
    analyzer = FaceAnalyzer(settings, settings.required_models())
    analyzer.check()
    source_faces, target_faces = analyzer.get_faces(source), analyzer.get_faces(target)
    assert len(source_faces) == len(target_faces) == 1
    assert np.isclose(np.linalg.norm(source_faces[0].embedding), 1)
    blank = analyzer.get_faces(np.zeros_like(source))
    assert blank == []
    swapper = FaceSwapper(root / "models/swap/inswapper_128.onnx", "cpu")
    swapper.check()
    converted = swapper.swap(
        source, source_faces[0], swapper.map_identity(target_faces[0].embedding)
    )
    assert converted.shape == source.shape
    assert np.mean(np.abs(converted.astype(float) - source.astype(float))) > 0.1


def test_real_enhancement_fixed_noise():
    import torch

    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    enhancer = FaceEnhancer(root / "models/enhance/GFPGANv1.4.pth", "cpu", 0.7)
    from media_tool.alignment import aligned_crop

    settings = Settings(
        root / "materials/input",
        root / "work",
        root / "materials/output",
        root / "faces/source",
        root / "faces/target",
        root / "models",
        onnx_provider="cpu",
        enhance_enabled=False,
    )
    analyzer = FaceAnalyzer(settings, settings.required_models())
    pixels = read_image(Path(os.environ["MEDIA_TOOL_TEST_FACE"])).frame
    face = analyzer.get_faces(pixels)[0]
    crop, _ = aligned_crop(pixels, face.kps, 512, enhance=True)
    first, second = enhancer.restore(crop), enhancer.restore(crop)
    assert first.shape == (512, 512, 3)
    assert np.array_equal(first, second)
