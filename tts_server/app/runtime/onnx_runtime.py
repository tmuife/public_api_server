from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

from app.runtime.config import RuntimeConfig
from app.utils.audio_encoding import float32_to_int16_pcm_bytes, waveform_to_wav_bytes
from app.utils.audio_processing import PreparedReferenceAudio, prepare_reference_audio

if TYPE_CHECKING:
    from app.runtime.moss_real_runtime import LocalMossOnnxRuntime


@dataclass(frozen=True)
class RuntimeMetadata:
    backend: str
    onnxruntime_available: bool
    onnxruntime_version: str | None
    model_dir: str | None


@dataclass(frozen=True)
class RuntimeSynthesisResult:
    waveform: np.ndarray  # [samples, channels], float32
    sample_rate: int
    metadata: RuntimeMetadata


class TorchlessOnnxRuntime:
    """Local runtime that keeps ONNX/TTS path torch-free.

    The runtime is intentionally independent from reference directories and
    from torch/torchaudio imports.
    """

    BUILTIN_VOICE_PROMPT_FILES: dict[str, str] = {
        "Junhao": "zh_1.wav",
        "Zhiming": "zh_2.wav",
        "Weiguo": "zh_5.wav",
        "Xiaoyu": "zh_3.wav",
        "Yuewen": "zh_4.wav",
        "Lingyu": "zh_6.wav",
        "Trump": "en_1.wav",
        "Ava": "en_2.wav",
        "Bella": "en_3.wav",
        "Adam": "en_4.wav",
        "Nathan": "en_5.wav",
        "Sakura": "jp_1.mp3",
        "Yui": "jp_2.wav",
        "Aoi": "jp_3.wav",
        "Hina": "jp_4.wav",
        "Mei": "jp_5.wav",
    }

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._initialized = False
        self._real_runtime: LocalMossOnnxRuntime | None = None
        self._real_runtime_error: str | None = None
        self._metadata = RuntimeMetadata(
            backend="onnxruntime-cpu-local",
            onnxruntime_available=False,
            onnxruntime_version=None,
            model_dir=str(config.model_dir) if config.model_dir else None,
        )

    def initialize(self) -> RuntimeMetadata:
        onnxruntime_available = False
        onnxruntime_version: str | None = None
        backend = "onnxruntime-cpu-local-fallback"

        try:
            import onnxruntime as ort  # noqa: PLC0415

            onnxruntime_available = True
            onnxruntime_version = str(getattr(ort, "__version__", "unknown"))
        except Exception:  # pragma: no cover - optional runtime dependency behavior
            logging.warning("onnxruntime is unavailable; runtime will use local fallback synthesis")

        model_dir: Path | None = None
        model_dir_exists = False
        if self.config.model_dir is not None:
            model_dir = Path(self.config.model_dir)
            if not model_dir.exists():
                logging.warning("Configured ONNX model directory does not exist: %s", model_dir)
            else:
                model_dir_exists = True

        self._real_runtime = None
        self._real_runtime_error = None
        if onnxruntime_available and model_dir_exists and model_dir is not None:
            try:
                from app.runtime.moss_real_runtime import LocalMossOnnxRuntime  # noqa: PLC0415

                self._real_runtime = LocalMossOnnxRuntime(model_dir=model_dir, config=self.config)
                backend = "onnxruntime-cpu-local-moss-real"
            except Exception as exc:  # pragma: no cover - runtime environment dependent
                self._real_runtime_error = str(exc)
                logging.exception("Failed to initialize local MOSS ONNX runtime; fallback synthesis will be used")

        self._metadata = RuntimeMetadata(
            backend=backend,
            onnxruntime_available=onnxruntime_available,
            onnxruntime_version=onnxruntime_version,
            model_dir=str(self.config.model_dir) if self.config.model_dir else None,
        )
        self._initialized = True
        return self._metadata

    def metadata(self) -> RuntimeMetadata:
        if not self._initialized:
            return self.initialize()
        return self._metadata

    def list_builtin_voices(self) -> list[str]:
        if not self._initialized:
            self.initialize()
        if self._real_runtime is not None:
            voices = [
                str(item.get("voice", "")).strip()
                for item in self._real_runtime.list_builtin_voices()
                if isinstance(item, dict)
            ]
            normalized = [voice for voice in voices if voice]
            if normalized:
                return normalized
        return list(self.BUILTIN_VOICE_PROMPT_FILES.keys())

    def resolve_builtin_voice_name(self, voice: str) -> str:
        normalized_voice = str(voice or "").strip()
        if not normalized_voice:
            raise ValueError("voice is required when reference audio is not provided")
        available_voices = self.list_builtin_voices()
        for candidate in available_voices:
            if candidate.lower() == normalized_voice.lower():
                return candidate
        supported = ", ".join(available_voices)
        raise ValueError(f"unsupported voice '{normalized_voice}'. Supported voices: {supported}")

    def get_builtin_voice_prompt_file(self, voice_name: str) -> str | None:
        resolved_voice = self.resolve_builtin_voice_name(voice_name)
        return self.BUILTIN_VOICE_PROMPT_FILES.get(resolved_voice)

    def resolve_builtin_voice_prompt_path(self, voice_name: str) -> Path | None:
        prompt_file = self.get_builtin_voice_prompt_file(voice_name)
        if not prompt_file or self.config.prompt_audio_dir is None:
            return None

        candidate = (self.config.prompt_audio_dir / prompt_file).expanduser().resolve()
        if not candidate.exists():
            return None
        return candidate

    def synthesize_to_wav_bytes(
        self,
        *,
        text: str,
        reference_audio_path: str | Path | None = None,
        voice: str | None = None,
    ) -> tuple[bytes, RuntimeMetadata]:
        result = self.synthesize_waveform(text=text, reference_audio_path=reference_audio_path, voice=voice)
        return waveform_to_wav_bytes(result.waveform, result.sample_rate), result.metadata

    def synthesize_waveform(
        self,
        *,
        text: str,
        reference_audio_path: str | Path | None = None,
        voice: str | None = None,
    ) -> RuntimeSynthesisResult:
        if not text or not str(text).strip():
            raise ValueError("text cannot be empty")
        metadata = self.metadata()

        if self._real_runtime is not None:
            resolved_voice = None
            if reference_audio_path is None or not str(reference_audio_path).strip():
                resolved_voice = self.resolve_builtin_voice_name(str(voice or ""))
            generated = self._real_runtime.synthesize_waveform(
                text=text,
                voice=resolved_voice,
                reference_audio_path=reference_audio_path,
                output_sample_rate=self.config.sample_rate,
                output_channels=self.config.channels,
            )
            return RuntimeSynthesisResult(
                waveform=generated.waveform,
                sample_rate=generated.sample_rate,
                metadata=metadata,
            )

        if reference_audio_path is not None and str(reference_audio_path).strip():
            prepared_reference = prepare_reference_audio(
                reference_audio_path,
                target_sample_rate=self.config.sample_rate,
                target_channels=self.config.channels,
            )
        else:
            prepared_reference = self._build_builtin_voice_reference(voice)

        waveform = self._synthesize_waveform_from_features(text, prepared_reference)
        return RuntimeSynthesisResult(waveform=waveform, sample_rate=self.config.sample_rate, metadata=metadata)

    def stream_pcm_chunks(
        self,
        *,
        text: str,
        reference_audio_path: str | Path | None = None,
        voice: str | None = None,
    ) -> tuple[Iterator[bytes], int, RuntimeMetadata]:
        result = self.synthesize_waveform(text=text, reference_audio_path=reference_audio_path, voice=voice)
        chunk_size = max(1, int(self.config.stream_chunk_samples))
        waveform = result.waveform

        def _generator() -> Iterator[bytes]:
            for start in range(0, waveform.shape[0], chunk_size):
                chunk = waveform[start : start + chunk_size, :]
                if chunk.size <= 0:
                    continue
                yield float32_to_int16_pcm_bytes(chunk)

        return _generator(), result.sample_rate, result.metadata

    def _build_builtin_voice_reference(self, voice: str | None) -> PreparedReferenceAudio:
        resolved_voice = self.resolve_builtin_voice_name(str(voice or ""))
        prompt_audio_path = self.resolve_builtin_voice_prompt_path(resolved_voice)
        if prompt_audio_path is not None:
            return prepare_reference_audio(
                prompt_audio_path,
                target_sample_rate=self.config.sample_rate,
                target_channels=self.config.channels,
            )

        sample_count = max(1, int(round(self.config.sample_rate * 1.0)))
        timeline = np.arange(sample_count, dtype=np.float32) / float(self.config.sample_rate)
        voice_seed = int(hashlib.sha1(resolved_voice.lower().encode("utf-8")).hexdigest()[:8], 16)

        fundamental_hz = 120.0 + float(voice_seed % 180)
        overtone_hz = fundamental_hz * (1.5 + float((voice_seed // 7) % 4) * 0.25)
        waveform = (
            0.2 * np.sin(2.0 * np.pi * fundamental_hz * timeline)
            + 0.08 * np.sin(2.0 * np.pi * overtone_hz * timeline)
        )
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)
        channel_major = waveform[:, None]
        if self.config.channels > 1:
            channel_major = np.repeat(channel_major, self.config.channels, axis=1)

        return PreparedReferenceAudio(waveform=channel_major, sample_rate=self.config.sample_rate)

    def _synthesize_waveform_from_features(self, text: str, reference_audio: PreparedReferenceAudio) -> np.ndarray:
        duration_seconds = self._estimate_duration_seconds(text)
        sample_count = max(1, int(round(duration_seconds * self.config.sample_rate)))
        time_axis = np.arange(sample_count, dtype=np.float32) / float(self.config.sample_rate)

        text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        text_seed = int(text_hash[:8], 16)

        reference_mono = np.mean(reference_audio.waveform, axis=1)
        rms = float(np.sqrt(np.mean(np.square(reference_mono)))) if reference_mono.size else 0.05
        centroid = self._estimate_spectral_centroid(reference_mono, reference_audio.sample_rate)

        fundamental_hz = 160.0 + float(text_seed % 170)
        harmonic_hz = max(fundamental_hz * 2.0, centroid * 0.5)
        modulation = 1.0 + min(0.45, rms)

        base = 0.24 * np.sin(2.0 * np.pi * fundamental_hz * time_axis)
        harmonic = 0.12 * np.sin(2.0 * np.pi * harmonic_hz * time_axis * modulation)

        envelope = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
        release = np.linspace(1.0, 0.0, min(2048, sample_count), dtype=np.float32)
        envelope[-release.shape[0] :] *= release

        mono_waveform = (base + harmonic) * envelope
        mono_waveform = np.clip(mono_waveform, -1.0, 1.0).astype(np.float32)
        waveform = mono_waveform[:, None]

        if self.config.channels > 1:
            waveform = np.repeat(waveform, self.config.channels, axis=1)

        return waveform.astype(np.float32, copy=False)

    def _estimate_duration_seconds(self, text: str) -> float:
        trimmed = str(text).strip()
        base_seconds = max(self.config.fallback_min_seconds, len(trimmed) * 0.055)
        return float(min(base_seconds, self.config.fallback_max_seconds))

    @staticmethod
    def _estimate_spectral_centroid(samples: np.ndarray, sample_rate: int) -> float:
        if samples.size <= 0 or sample_rate <= 0:
            return 220.0

        window = np.asarray(samples, dtype=np.float32)
        window = window[: min(window.shape[0], sample_rate * 2)]
        if window.size <= 1:
            return 220.0

        spectrum = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(window.shape[0], d=1.0 / float(sample_rate))
        total = float(np.sum(spectrum))
        if total <= 1e-9:
            return 220.0
        return float(np.sum(freqs * spectrum) / total)
