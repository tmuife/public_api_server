from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

from app.runtime.onnx_runtime import TorchlessOnnxRuntime
from app.schemas.openai_compatible import (
    OpenAiModelCard,
    OpenAiModelListResponse,
    OpenAiVoiceCard,
    OpenAiVoiceListResponse,
)
from app.services.openai_mappings import OpenAiMappings, load_openai_mappings
from app.services.tts_service import TtsService
from app.utils.audio_encoding import float32_to_int16_pcm_bytes, waveform_to_wav_bytes

MAX_REFERENCE_AUDIO_BYTES = 8 * 1024 * 1024
DEFAULT_STREAM_CHUNK_BYTES = 4096


@dataclass(frozen=True)
class OpenAiSpeechSynthesisResult:
    audio_bytes: bytes
    sample_rate: int
    response_format: str
    source: str

    @property
    def media_type(self) -> str:
        return "audio/wav" if self.response_format == "wav" else "audio/pcm"


@dataclass(frozen=True)
class ReferenceAudioResolution:
    reference_audio_path: str | None
    source: str
    cleanup_paths: tuple[Path, ...]


class OpenAiSpeechService:
    def __init__(self, runtime: TorchlessOnnxRuntime, mappings: OpenAiMappings) -> None:
        self.runtime = runtime
        self.mappings = mappings

    def list_models(self) -> OpenAiModelListResponse:
        created = int(time.time())
        data = [OpenAiModelCard(id=model_id, created=created) for model_id in self.mappings.public_model_ids]
        return OpenAiModelListResponse(data=data)

    def list_voices(self) -> OpenAiVoiceListResponse:
        aliases_by_canonical_voice: dict[str, list[str]] = {}
        for alias, canonical_voice in self.mappings.voice_alias_lookup.items():
            aliases = aliases_by_canonical_voice.setdefault(canonical_voice, [])
            if alias not in aliases:
                aliases.append(alias)

        cards: list[OpenAiVoiceCard] = []
        for canonical_voice in self.runtime.list_builtin_voices():
            prompt_audio_file = self.runtime.get_builtin_voice_prompt_file(canonical_voice)
            prompt_audio_path = self.runtime.resolve_builtin_voice_prompt_path(canonical_voice)
            cards.append(
                OpenAiVoiceCard(
                    canonical=canonical_voice,
                    aliases=sorted(aliases_by_canonical_voice.get(canonical_voice, [])),
                    prompt_audio_file=prompt_audio_file,
                    prompt_audio_path=str(prompt_audio_path) if prompt_audio_path is not None else None,
                    prompt_audio_configured=self.runtime.config.prompt_audio_dir is not None,
                    prompt_audio_exists=prompt_audio_path is not None,
                )
            )
        return OpenAiVoiceListResponse(data=cards)

    def synthesize_audio(
        self,
        *,
        model: str,
        input_text: str,
        voice: str,
        response_format: str,
        reference_audio_path: str | None,
        reference_audio_url: str | None,
        reference_audio_file_bytes: bytes | None,
        reference_audio_file_name: str | None,
    ) -> OpenAiSpeechSynthesisResult:
        self._resolve_model(model)

        normalized_text = str(input_text or "").strip()
        if not normalized_text:
            raise ValueError("input cannot be empty")

        normalized_format = str(response_format or "").strip().lower()
        if normalized_format not in {"wav", "pcm"}:
            raise ValueError("response_format must be one of: wav, pcm")

        resolution = self._resolve_reference_audio(
            reference_audio_path=reference_audio_path,
            reference_audio_url=reference_audio_url,
            reference_audio_file_bytes=reference_audio_file_bytes,
            reference_audio_file_name=reference_audio_file_name,
        )
        resolved_voice = self._resolve_builtin_voice(voice) if resolution.source == "voice" else str(voice or "").strip()

        try:
            synthesis_result = self.runtime.synthesize_waveform(
                text=normalized_text,
                reference_audio_path=resolution.reference_audio_path,
                voice=resolved_voice,
            )

            waveform = synthesis_result.waveform
            if normalized_format == "wav":
                encoded = waveform_to_wav_bytes(waveform, synthesis_result.sample_rate)
            else:
                encoded = float32_to_int16_pcm_bytes(waveform)

            return OpenAiSpeechSynthesisResult(
                audio_bytes=encoded,
                sample_rate=synthesis_result.sample_rate,
                response_format=normalized_format,
                source=resolution.source,
            )
        finally:
            for cleanup_path in resolution.cleanup_paths:
                cleanup_path.unlink(missing_ok=True)

    def chunk_audio_bytes(self, audio_bytes: bytes, chunk_size: int = DEFAULT_STREAM_CHUNK_BYTES) -> Iterable[bytes]:
        effective_size = max(1, int(chunk_size))
        for index in range(0, len(audio_bytes), effective_size):
            chunk = audio_bytes[index : index + effective_size]
            if chunk:
                yield chunk

    def _resolve_reference_audio(
        self,
        *,
        reference_audio_path: str | None,
        reference_audio_url: str | None,
        reference_audio_file_bytes: bytes | None,
        reference_audio_file_name: str | None,
    ) -> ReferenceAudioResolution:
        cleanup_paths: list[Path] = []
        if reference_audio_file_bytes is not None:
            temp_path = self._write_temp_audio(
                audio_bytes=reference_audio_file_bytes,
                source_name=reference_audio_file_name or "reference_audio_file",
            )
            cleanup_paths.append(temp_path)
            return ReferenceAudioResolution(
                reference_audio_path=str(temp_path),
                source="reference_audio_file",
                cleanup_paths=tuple(cleanup_paths),
            )

        if reference_audio_url:
            downloaded = self._download_audio_from_url(reference_audio_url)
            cleanup_paths.append(downloaded)
            return ReferenceAudioResolution(
                reference_audio_path=str(downloaded),
                source="reference_audio_url",
                cleanup_paths=tuple(cleanup_paths),
            )

        normalized_path = str(reference_audio_path or "").strip()
        if normalized_path:
            resolved = Path(normalized_path).expanduser().resolve()
            return ReferenceAudioResolution(
                reference_audio_path=str(resolved),
                source="reference_audio_path",
                cleanup_paths=tuple(cleanup_paths),
            )

        return ReferenceAudioResolution(
            reference_audio_path=None,
            source="voice",
            cleanup_paths=tuple(cleanup_paths),
        )

    def _resolve_model(self, model: str) -> str:
        normalized_model = str(model or "").strip()
        if not normalized_model:
            raise ValueError("model is required")
        resolved = self.mappings.model_alias_lookup.get(normalized_model.lower())
        if resolved is not None:
            return resolved
        supported_models = ", ".join(self.mappings.public_model_ids)
        raise ValueError(f"unsupported model '{normalized_model}'. Supported models: {supported_models}")

    def _resolve_builtin_voice(self, voice: str) -> str:
        normalized_voice = str(voice or "").strip()
        if not normalized_voice:
            raise ValueError("voice is required when reference audio is not provided")

        resolved = self.mappings.voice_alias_lookup.get(normalized_voice.lower(), normalized_voice)
        try:
            return self.runtime.resolve_builtin_voice_name(resolved)
        except ValueError as exc:
            aliases = ", ".join(self.mappings.public_voice_aliases)
            canonical_voices = ", ".join(self.runtime.list_builtin_voices())
            raise ValueError(
                f"unsupported voice '{normalized_voice}'. "
                f"Supported aliases: {aliases}. Supported canonical voices: {canonical_voices}"
            ) from exc

    def _download_audio_from_url(self, source_url: str) -> Path:
        parsed = urlparse(source_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("reference_audio_url must use http or https")

        with urlopen(source_url, timeout=8) as response:  # noqa: S310
            payload = response.read(MAX_REFERENCE_AUDIO_BYTES + 1)
        if len(payload) > MAX_REFERENCE_AUDIO_BYTES:
            raise ValueError("reference_audio_url payload exceeds 8MB limit")
        return self._write_temp_audio(audio_bytes=payload, source_name="reference_audio_url")

    def _write_temp_audio(self, *, audio_bytes: bytes, source_name: str) -> Path:
        if len(audio_bytes) <= 0:
            raise ValueError(f"{source_name} is empty")
        if len(audio_bytes) > MAX_REFERENCE_AUDIO_BYTES:
            raise ValueError(f"{source_name} exceeds 8MB limit")

        suffix = ".wav"
        with tempfile.NamedTemporaryFile(prefix="tts-ref-", suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            temp_path = Path(tmp_file.name).resolve()
        return temp_path


def build_openai_speech_service(tts_service: TtsService) -> OpenAiSpeechService:
    return OpenAiSpeechService(runtime=tts_service.runtime, mappings=load_openai_mappings())
