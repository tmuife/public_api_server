from pathlib import Path

import pytest

from media_tool.config import Settings


@pytest.fixture
def settings(tmp_path):
    roots = {
        name: tmp_path / name
        for name in ("input", "work", "output", "reference", "target", "models")
    }
    for path in roots.values():
        path.mkdir()
    return Settings(
        roots["input"],
        roots["work"],
        roots["output"],
        roots["reference"],
        roots["target"],
        roots["models"],
        enhance_enabled=False,
        onnx_provider="cpu",
        enhance_device="cpu",
        video_preset="ultrafast",
    )


@pytest.fixture
def env_file(settings, tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "\n".join(
            f"{key}={value}"
            for key, value in settings.report_dict().items()
            if isinstance(getattr(settings, key), Path)
        ),
        encoding="utf-8",
    )
    return path
