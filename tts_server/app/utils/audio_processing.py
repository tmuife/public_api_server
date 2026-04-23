from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


@dataclass(frozen=True)
class PreparedReferenceAudio:
    """Reference audio after torchless loading and preprocessing."""

    waveform: np.ndarray  # shape: [samples, channels], float32
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.waveform.shape[1])

    @property
    def samples(self) -> int:
        return int(self.waveform.shape[0])

    def to_model_input(self) -> np.ndarray:
        """Convert to [1, channels, samples] float32 layout."""
        channel_major = np.transpose(self.waveform, (1, 0))
        return channel_major[None, :, :].astype(np.float32, copy=False)


def load_reference_audio(path: str | Path) -> tuple[np.ndarray, int]:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {resolved_path}")

    waveform, sample_rate = sf.read(str(resolved_path), dtype="float32", always_2d=True)
    if waveform.size <= 0:
        raise ValueError(f"Reference audio is empty: {resolved_path}")

    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    normalized = np.asarray(waveform, dtype=np.float32)
    if normalized.ndim != 2:
        raise ValueError(f"Expected 2D waveform [samples, channels], got {normalized.shape}")

    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 1.0:
        normalized = normalized / peak

    return normalized.astype(np.float32, copy=False)


def convert_channels(waveform: np.ndarray, target_channels: int) -> np.ndarray:
    if target_channels <= 0:
        raise ValueError("target_channels must be positive")

    current_channels = int(waveform.shape[1])
    if current_channels == target_channels:
        return waveform

    if current_channels == 1 and target_channels > 1:
        return np.repeat(waveform, target_channels, axis=1)

    if current_channels > 1 and target_channels == 1:
        return np.mean(waveform, axis=1, keepdims=True).astype(np.float32)

    raise ValueError(f"Unsupported channel conversion: {current_channels} -> {target_channels}")


def resample_audio_soxr(waveform: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("source_sample_rate and target_sample_rate must be positive")

    if source_sample_rate == target_sample_rate:
        return waveform

    # soxr is the primary backend for torchless high-quality resampling.
    channels = int(waveform.shape[1])
    resampled_channels: list[np.ndarray] = []
    for channel_index in range(channels):
        channel = np.asarray(waveform[:, channel_index], dtype=np.float32)
        resampled = soxr.resample(channel, source_sample_rate, target_sample_rate, quality="HQ")
        resampled_channels.append(np.asarray(resampled, dtype=np.float32))

    min_length = min(len(channel) for channel in resampled_channels)
    stacked = np.stack([channel[:min_length] for channel in resampled_channels], axis=1)
    return stacked.astype(np.float32, copy=False)


def prepare_reference_audio(
    path: str | Path,
    *,
    target_sample_rate: int,
    target_channels: int,
) -> PreparedReferenceAudio:
    waveform, sample_rate = load_reference_audio(path)
    waveform = normalize_audio(waveform)
    waveform = resample_audio_soxr(waveform, sample_rate, target_sample_rate)
    waveform = convert_channels(waveform, target_channels)
    waveform = normalize_audio(waveform)

    if waveform.shape[0] <= 0:
        raise ValueError("Reference audio preprocessing produced empty waveform")

    return PreparedReferenceAudio(waveform=waveform, sample_rate=target_sample_rate)
