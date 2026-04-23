from __future__ import annotations

import io
import wave

import numpy as np


def float32_to_int16_pcm_bytes(waveform: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    int16_audio = (clipped * 32767.0).astype(np.int16)
    return int16_audio.tobytes(order="C")


def waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    wave_array = np.asarray(waveform, dtype=np.float32)
    if wave_array.ndim == 1:
        wave_array = wave_array[:, None]
    if wave_array.ndim != 2:
        raise ValueError(f"Expected waveform shape [samples, channels], got {wave_array.shape}")

    channels = int(wave_array.shape[1])
    pcm_bytes = float32_to_int16_pcm_bytes(wave_array)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)

    return buffer.getvalue()
