from __future__ import annotations

import os
from pathlib import Path

from decouple import Config, RepositoryEnv

_SUPPORTED_ENV_KEYS = (
    "APP_HOST",
    "APP_PORT",
    "AUTH_REQUIRED",
    "DOCS_PUBLIC_IN_DEV",
    "API_KEY",
    "ONNX_MODEL_DIR",
    "TTS_PROMPT_AUDIO_DIR",
    "TTS_SAMPLE_RATE",
    "TTS_CHANNELS",
    "TTS_STREAM_CHUNK_SAMPLES",
    "ONNX_CPU_THREADS",
    "ONNX_MAX_NEW_FRAMES",
    "ONNX_DO_SAMPLE",
    "ONNX_SAMPLE_MODE",
    "ONNX_SEED",
    "ONNX_VOICE_CLONE_MAX_TEXT_TOKENS",
    "ONNX_TEXT_TEMPERATURE",
    "ONNX_TEXT_TOP_P",
    "ONNX_TEXT_TOP_K",
    "ONNX_AUDIO_TEMPERATURE",
    "ONNX_AUDIO_TOP_P",
    "ONNX_AUDIO_TOP_K",
    "ONNX_AUDIO_REPETITION_PENALTY",
    "ONNX_ENABLE_NORMALIZE_TTS_TEXT",
)


def load_dotenv_into_os_environ(dotenv_path: Path | None = None) -> None:
    """Load .env values into os.environ without overriding existing env vars."""
    if dotenv_path is None:
        project_root = Path(__file__).resolve().parents[2]
        dotenv_path = project_root / ".env"

    resolved_path = dotenv_path.resolve()
    if not resolved_path.exists():
        return

    config = Config(RepositoryEnv(str(resolved_path)))
    for key in _SUPPORTED_ENV_KEYS:
        if key in os.environ:
            continue
        value = config(key, default=None)
        if value is None:
            continue
        os.environ[key] = str(value)
