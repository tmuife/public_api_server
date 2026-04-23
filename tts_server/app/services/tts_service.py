from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.runtime.config import RuntimeConfig
from app.runtime.onnx_runtime import RuntimeMetadata, TorchlessOnnxRuntime


@dataclass(frozen=True)
class BatchSynthesisOutput:
    audio_bytes: bytes
    sample_rate: int
    metadata: RuntimeMetadata


@dataclass(frozen=True)
class StreamSynthesisOutput:
    chunk_bytes: list[bytes]
    aggregated_audio_bytes: bytes
    sample_rate: int
    metadata: RuntimeMetadata


class TtsService:
    def __init__(self, runtime: TorchlessOnnxRuntime) -> None:
        self.runtime = runtime

    def initialize(self) -> RuntimeMetadata:
        return self.runtime.initialize()

    def synthesize_batch(self, *, text: str, reference_audio_path: str | Path) -> BatchSynthesisOutput:
        audio_bytes, metadata = self.runtime.synthesize_to_wav_bytes(
            text=text,
            reference_audio_path=reference_audio_path,
        )
        return BatchSynthesisOutput(
            audio_bytes=audio_bytes,
            sample_rate=self.runtime.config.sample_rate,
            metadata=metadata,
        )

    def synthesize_stream(self, *, text: str, reference_audio_path: str | Path) -> StreamSynthesisOutput:
        iterator, sample_rate, metadata = self.runtime.stream_pcm_chunks(
            text=text,
            reference_audio_path=reference_audio_path,
        )
        chunks = list(iterator)
        aggregated = b"".join(chunks)
        return StreamSynthesisOutput(
            chunk_bytes=chunks,
            aggregated_audio_bytes=aggregated,
            sample_rate=sample_rate,
            metadata=metadata,
        )


@lru_cache(maxsize=1)
def build_tts_service() -> TtsService:
    config = RuntimeConfig.from_env()
    runtime = TorchlessOnnxRuntime(config)
    return TtsService(runtime)
