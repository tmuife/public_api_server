from types import SimpleNamespace

import numpy as np
import pytest

from media_tool.alignment import ARCFACE_TEMPLATE, similarity_matrix
from media_tool.errors import MediaToolError
from media_tool.face_matcher import FaceMatcher, register_faces
from media_tool.faces import DetectedFace, normalize_embedding
from media_tool.pipeline import FacePipeline


def feature(index):
    array = np.zeros(512, np.float32)
    array[index] = 1
    return array


def face(index):
    return DetectedFace(
        np.array([0, 0, 30, 30], np.float32), np.zeros((5, 2), np.float32), 0.9, feature(index)
    )


def test_matching_any_reference_once_and_strict_boundary():
    one, two, other = face(0), face(1), face(2)
    matcher = FaceMatcher([("a", feature(0)), ("b", feature(1)), ("a2", feature(0))], 1.25)
    matches = matcher.match([one, two, other])
    assert [m.face for m in matches] == [one, two]
    assert [m.distance for m in matches] == [0, 0]
    boundary = float(np.linalg.norm(feature(0) - feature(1)))
    assert FaceMatcher([("a", feature(0))], boundary).match([two]) == []


@pytest.mark.parametrize("value", [np.zeros(512), np.ones(511), np.full(512, float("inf"))])
def test_invalid_features(value):
    with pytest.raises(MediaToolError):
        normalize_embedding(value)


def test_all_matches_swapped_before_enhancement():
    faces, calls = [face(0), face(1), face(2)], []
    matcher = FaceMatcher([("a", feature(0)), ("b", feature(1))], 1.25)

    class Swapper:
        def swap(self, frame, detected, latent):
            calls.append(("swap", detected))
            return frame + 1

    class Enhancer:
        def enhance(self, frame, detected):
            calls.append(("enhance", detected))
            return frame + 1

    pipeline = FacePipeline(
        SimpleNamespace(get_faces=lambda f: faces), matcher, Swapper(), None, Enhancer()
    )
    result = pipeline.process(np.zeros((10, 10, 3), np.uint8))
    assert [kind for kind, _ in calls] == ["swap", "swap", "enhance", "enhance"]
    assert result.swapped == result.enhanced == 2
    assert np.all(result.frame == 4)


def test_unmatched_reuses_pixels_without_inference():
    original = np.zeros((10, 10, 3), np.uint8)
    matcher = FaceMatcher([("reference", feature(0))], 1.25)
    pipeline = FacePipeline(SimpleNamespace(get_faces=lambda f: [face(1)]), matcher, None, None)
    assert pipeline.process(original).frame is original


def test_enhancement_failure_propagates():
    class Enhancer:
        def enhance(self, frame, face):
            raise MediaToolError("增强失败")

    pipeline = FacePipeline(
        SimpleNamespace(get_faces=lambda f: [face(0)]),
        FaceMatcher([("a", feature(0))], 1.25),
        SimpleNamespace(swap=lambda frame, *args: frame),
        None,
        Enhancer(),
    )
    with pytest.raises(MediaToolError, match="增强失败"):
        pipeline.process(np.zeros((10, 10, 3), np.uint8))


def test_alignment_recovers_known_similarity():
    angle = 0.31
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    source = (ARCFACE_TEMPLATE - [15, 8]) @ rotation / 1.7
    matrix = similarity_matrix(source, ARCFACE_TEMPLATE)
    assert np.allclose(source @ matrix[:, :2].T + matrix[:, 2], ARCFACE_TEMPLATE, atol=1e-8)


def test_empty_reference_and_multiple_target(settings):
    with pytest.raises(MediaToolError, match="参考图片目录为空"):
        register_faces(settings, None)
