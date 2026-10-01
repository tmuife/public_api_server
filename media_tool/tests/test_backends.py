from pathlib import Path

import numpy as np
import pytest

from media_tool.backends import OnnxModel
from media_tool.errors import MediaToolError


def test_explicit_coreml_unavailable(monkeypatch):
    monkeypatch.setattr(
        "media_tool.backends.ort.get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    with pytest.raises(MediaToolError, match="不支持"):
        OnnxModel(Path("unused.onnx"), "coreml")


def test_auto_backend_fallback_is_per_model_and_reported(monkeypatch, caplog):
    calls = []

    class Session:
        def __init__(self, path, sess_options, providers):
            self.providers = providers
            calls.append(providers)

        def disable_fallback(self):
            pass

        def get_providers(self):
            return self.providers

        def run(self, outputs, feeds):
            if "CoreMLExecutionProvider" in self.providers:
                raise RuntimeError("CoreML unsupported operator")
            return [np.zeros(1)]

    monkeypatch.setattr(
        "media_tool.backends.ort.get_available_providers",
        lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr("media_tool.backends.ort.InferenceSession", Session)
    model = OnnxModel(Path("recognition.onnx"), "auto")
    model.run({})
    assert len(calls) == 2
    assert model.describe()["providers"] == ["CPUExecutionProvider"]
    assert "recognition.onnx" in caplog.text and "CPU" in caplog.text


def test_invalid_model_not_treated_as_backend_failure(monkeypatch):
    calls = []

    def invalid(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("invalid protobuf")

    monkeypatch.setattr(
        "media_tool.backends.ort.get_available_providers",
        lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr("media_tool.backends.ort.InferenceSession", invalid)
    with pytest.raises(MediaToolError, match="invalid protobuf"):
        OnnxModel(Path("broken.onnx"), "auto")
    assert len(calls) == 1


def test_explicit_mps_unavailable(monkeypatch):
    from media_tool.face_enhancer import FaceEnhancer

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    with pytest.raises(MediaToolError, match="不支持"):
        FaceEnhancer(Path("unused.pth"), "mps", 0.7)
