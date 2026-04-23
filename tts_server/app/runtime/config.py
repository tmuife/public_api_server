from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings for the local torchless ONNX path."""

    model_dir: Path | None
    prompt_audio_dir: Path | None
    sample_rate: int = 16000
    channels: int = 1
    stream_chunk_samples: int = 3200
    fallback_min_seconds: float = 0.8
    fallback_max_seconds: float = 12.0
    onnx_cpu_threads: int = 4
    onnx_max_new_frames: int | None = None
    onnx_do_sample: bool | None = None
    onnx_sample_mode: str | None = None
    onnx_seed: int | None = 0
    onnx_voice_clone_max_text_tokens: int = 75
    onnx_text_temperature: float | None = None
    onnx_text_top_p: float | None = None
    onnx_text_top_k: int | None = None
    onnx_audio_temperature: float | None = None
    onnx_audio_top_p: float | None = None
    onnx_audio_top_k: int | None = None
    onnx_audio_repetition_penalty: float | None = None
    onnx_enable_normalize_tts_text: bool = True

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        project_root = Path(__file__).resolve().parents[2]
        default_model_dir = (project_root / "models").resolve()

        model_dir_raw = os.getenv("ONNX_MODEL_DIR", "").strip()
        if model_dir_raw:
            model_dir = Path(model_dir_raw).expanduser().resolve()
        elif default_model_dir.exists():
            model_dir = default_model_dir
        else:
            model_dir = None

        default_prompt_audio_dir = (project_root / "assets" / "audio").resolve()

        prompt_audio_dir_raw = os.getenv("TTS_PROMPT_AUDIO_DIR", "").strip()
        if prompt_audio_dir_raw:
            prompt_audio_dir = Path(prompt_audio_dir_raw).expanduser().resolve()
        elif default_prompt_audio_dir.exists():
            prompt_audio_dir = default_prompt_audio_dir
        else:
            prompt_audio_dir = None

        sample_rate = _read_int_env("TTS_SAMPLE_RATE", 16000, minimum=8000)
        channels = _read_int_env("TTS_CHANNELS", 1, minimum=1)
        stream_chunk_samples = _read_int_env("TTS_STREAM_CHUNK_SAMPLES", 3200, minimum=512)
        onnx_cpu_threads = _read_int_env("ONNX_CPU_THREADS", 4, minimum=1)
        onnx_max_new_frames = _read_optional_int_env("ONNX_MAX_NEW_FRAMES", minimum=1)
        onnx_do_sample = _read_optional_bool_env("ONNX_DO_SAMPLE")
        onnx_sample_mode = _read_optional_sample_mode_env("ONNX_SAMPLE_MODE")
        onnx_seed = _read_optional_int_env("ONNX_SEED")
        if onnx_seed is None:
            onnx_seed = 0
        onnx_voice_clone_max_text_tokens = _read_int_env("ONNX_VOICE_CLONE_MAX_TEXT_TOKENS", 75, minimum=1)
        onnx_text_temperature = _read_optional_float_env("ONNX_TEXT_TEMPERATURE", minimum=0.000001)
        onnx_text_top_p = _read_optional_float_env("ONNX_TEXT_TOP_P", minimum=0.000001)
        onnx_text_top_k = _read_optional_int_env("ONNX_TEXT_TOP_K", minimum=1)
        onnx_audio_temperature = _read_optional_float_env("ONNX_AUDIO_TEMPERATURE", minimum=0.000001)
        onnx_audio_top_p = _read_optional_float_env("ONNX_AUDIO_TOP_P", minimum=0.000001)
        onnx_audio_top_k = _read_optional_int_env("ONNX_AUDIO_TOP_K", minimum=1)
        onnx_audio_repetition_penalty = _read_optional_float_env("ONNX_AUDIO_REPETITION_PENALTY", minimum=0.000001)
        onnx_enable_normalize_tts_text = _read_bool_env("ONNX_ENABLE_NORMALIZE_TTS_TEXT", default=True)

        return cls(
            model_dir=model_dir,
            prompt_audio_dir=prompt_audio_dir,
            sample_rate=sample_rate,
            channels=channels,
            stream_chunk_samples=stream_chunk_samples,
            onnx_cpu_threads=onnx_cpu_threads,
            onnx_max_new_frames=onnx_max_new_frames,
            onnx_do_sample=onnx_do_sample,
            onnx_sample_mode=onnx_sample_mode,
            onnx_seed=onnx_seed,
            onnx_voice_clone_max_text_tokens=onnx_voice_clone_max_text_tokens,
            onnx_text_temperature=onnx_text_temperature,
            onnx_text_top_p=onnx_text_top_p,
            onnx_text_top_k=onnx_text_top_k,
            onnx_audio_temperature=onnx_audio_temperature,
            onnx_audio_top_p=onnx_audio_top_p,
            onnx_audio_top_k=onnx_audio_top_k,
            onnx_audio_repetition_penalty=onnx_audio_repetition_penalty,
            onnx_enable_normalize_tts_text=onnx_enable_normalize_tts_text,
        )


def _read_int_env(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _read_bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _read_optional_int_env(name: str, *, minimum: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = str(raw).strip()
    if not normalized:
        return None
    try:
        value = int(normalized)
    except ValueError:
        return None
    if minimum is not None and value < minimum:
        return None
    return value


def _read_optional_float_env(name: str, *, minimum: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = str(raw).strip()
    if not normalized:
        return None
    try:
        value = float(normalized)
    except ValueError:
        return None
    if minimum is not None and value < minimum:
        return None
    return value


def _read_optional_bool_env(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _read_optional_sample_mode_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"greedy", "fixed", "full"}:
        return normalized
    return None
