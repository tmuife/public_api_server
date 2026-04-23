from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm

from app.runtime.config import RuntimeConfig
from app.runtime.moss_ort_cpu_runtime import OrtCpuRuntime
from app.utils.audio_processing import (
    convert_channels,
    normalize_audio,
    prepare_reference_audio,
    resample_audio_soxr,
)

SENTENCE_END_PUNCTUATION = set(".!?。！？；;")
CLAUSE_SPLIT_PUNCTUATION = set(",，、；;：:")
CLOSING_PUNCTUATION = set("\"'”’)]}）】》」』")
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS = 0.40
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS = 0.24


@dataclass(frozen=True)
class MossSynthesisResult:
    waveform: np.ndarray  # [samples, channels], float32
    sample_rate: int


class LocalMossOnnxRuntime:
    """Local wrapper around copied MOSS ONNX CPU runtime core.

    This wrapper intentionally stays independent from external reference runtime imports.
    """

    def __init__(self, *, model_dir: str | Path, config: RuntimeConfig) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.config = config
        self.runtime = OrtCpuRuntime(
            model_dir=self.model_dir,
            thread_count=int(self.config.onnx_cpu_threads),
            max_new_frames=self.config.onnx_max_new_frames,
            do_sample=self.config.onnx_do_sample,
            sample_mode=self.config.onnx_sample_mode,
        )
        self._apply_generation_defaults()

        tokenizer_relative_path = str(self.runtime.manifest["model_files"].get("tokenizer_model", "tokenizer.model"))
        tokenizer_path = self.runtime.resolve_manifest_relative_path(tokenizer_relative_path)
        self.sp_model = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

        self.codec_sample_rate = int(self.runtime.codec_meta["codec_config"]["sample_rate"])
        self.codec_channels = int(self.runtime.codec_meta["codec_config"]["channels"])

    def list_builtin_voices(self) -> list[dict[str, object]]:
        return list(self.runtime.list_builtin_voices())

    def synthesize_waveform(
        self,
        *,
        text: str,
        voice: str | None,
        reference_audio_path: str | Path | None,
        output_sample_rate: int,
        output_channels: int,
    ) -> MossSynthesisResult:
        normalized_text = (
            self._prepare_text_for_inference(text)
            if self.config.onnx_enable_normalize_tts_text
            else str(text or "").strip()
        )
        if not normalized_text:
            raise ValueError("text cannot be empty")
        self._reset_rng_for_request()

        prompt_audio_codes = self._resolve_prompt_audio_codes(voice=voice, reference_audio_path=reference_audio_path)
        text_chunks = self._split_voice_clone_text(
            normalized_text,
            max_tokens=int(self.config.onnx_voice_clone_max_text_tokens),
        )

        generated_chunk_waveforms: list[np.ndarray] = []
        for chunk_index, chunk_text in enumerate(text_chunks):
            text_token_ids = [int(token_id) for token_id in self.sp_model.encode(chunk_text, out_type=int)]
            if not text_token_ids:
                continue
            request_rows = self.runtime.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids)
            generated_frames = self.runtime.generate_audio_frames(request_rows)
            channel_arrays, audio_length = self.runtime.decode_full_audio(generated_frames)
            waveform_chunk = self._merge_audio_channels(channel_arrays, audio_length)
            if waveform_chunk.size > 0:
                generated_chunk_waveforms.append(waveform_chunk)
            if chunk_index < len(text_chunks) - 1:
                pause_seconds = self._estimate_voice_clone_inter_chunk_pause_seconds(chunk_text)
                pause_samples = max(0, int(round(self.codec_sample_rate * pause_seconds)))
                if pause_samples > 0:
                    generated_chunk_waveforms.append(
                        np.zeros((pause_samples, int(self.codec_channels)), dtype=np.float32)
                    )
        waveform = self._concat_waveforms(generated_chunk_waveforms)

        processed = normalize_audio(waveform)
        if self.codec_sample_rate != int(output_sample_rate):
            processed = resample_audio_soxr(processed, self.codec_sample_rate, int(output_sample_rate))
        processed = convert_channels(processed, int(output_channels))
        processed = normalize_audio(processed)

        return MossSynthesisResult(
            waveform=processed.astype(np.float32, copy=False),
            sample_rate=int(output_sample_rate),
        )

    def _apply_generation_defaults(self) -> None:
        generation_defaults = self.runtime.manifest["generation_defaults"]
        if self.config.onnx_text_temperature is not None:
            generation_defaults["text_temperature"] = float(self.config.onnx_text_temperature)
        if self.config.onnx_text_top_p is not None:
            generation_defaults["text_top_p"] = float(self.config.onnx_text_top_p)
        if self.config.onnx_text_top_k is not None:
            generation_defaults["text_top_k"] = int(self.config.onnx_text_top_k)
        if self.config.onnx_audio_temperature is not None:
            generation_defaults["audio_temperature"] = float(self.config.onnx_audio_temperature)
        if self.config.onnx_audio_top_p is not None:
            generation_defaults["audio_top_p"] = float(self.config.onnx_audio_top_p)
        if self.config.onnx_audio_top_k is not None:
            generation_defaults["audio_top_k"] = int(self.config.onnx_audio_top_k)
        if self.config.onnx_audio_repetition_penalty is not None:
            generation_defaults["audio_repetition_penalty"] = float(self.config.onnx_audio_repetition_penalty)

    def _reset_rng_for_request(self) -> None:
        if self.config.onnx_seed is None:
            return
        self.runtime.rng = np.random.default_rng(int(self.config.onnx_seed))

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        for character in str(text or ""):
            if (
                "\u4e00" <= character <= "\u9fff"
                or "\u3400" <= character <= "\u4dbf"
                or "\u3040" <= character <= "\u30ff"
                or "\uac00" <= character <= "\ud7af"
            ):
                return True
        return False

    @classmethod
    def _prepare_text_for_inference(cls, text: str) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return ""
        normalized = normalized.replace("\r", " ").replace("\n", " ")
        while "  " in normalized:
            normalized = normalized.replace("  ", " ")
        if cls._contains_cjk(normalized):
            if normalized[-1] not in SENTENCE_END_PUNCTUATION:
                normalized += "。"
            return normalized
        if normalized[:1].islower():
            normalized = normalized[:1].upper() + normalized[1:]
        if normalized[-1].isalnum():
            normalized += "."
        if len([item for item in normalized.split() if item]) < 5:
            normalized = f"        {normalized}"
        return normalized

    def _count_text_tokens(self, text: str) -> int:
        return len(self.sp_model.encode(str(text or ""), out_type=int))

    def _split_text_by_token_budget(self, text: str, max_tokens: int) -> list[str]:
        remaining_text = str(text or "").strip()
        if not remaining_text:
            return []
        pieces: list[str] = []
        preferred_boundary_chars = set(CLAUSE_SPLIT_PUNCTUATION) | set(SENTENCE_END_PUNCTUATION) | {" "}
        while remaining_text:
            if self._count_text_tokens(remaining_text) <= max_tokens:
                pieces.append(remaining_text)
                break
            low = 1
            high = len(remaining_text)
            best_prefix_length = 1
            while low <= high:
                middle = (low + high) // 2
                candidate = remaining_text[:middle].strip()
                if not candidate:
                    low = middle + 1
                    continue
                if self._count_text_tokens(candidate) <= max_tokens:
                    best_prefix_length = middle
                    low = middle + 1
                else:
                    high = middle - 1
            cut_index = best_prefix_length
            prefix = remaining_text[:best_prefix_length]
            preferred_index = -1
            scan_min = max(-1, len(prefix) - 25)
            for scan_index in range(len(prefix) - 1, scan_min, -1):
                if prefix[scan_index] in preferred_boundary_chars:
                    preferred_index = scan_index + 1
                    break
            if preferred_index > 0:
                cut_index = preferred_index
            piece = remaining_text[:cut_index].strip()
            if not piece:
                piece = remaining_text[:best_prefix_length].strip()
                cut_index = best_prefix_length
            pieces.append(piece)
            remaining_text = remaining_text[cut_index:].strip()
        return pieces

    @staticmethod
    def _split_text_by_punctuation(text: str, punctuation: set[str]) -> list[str]:
        sentences: list[str] = []
        current_chars: list[str] = []
        index = 0
        normalized_text = str(text or "")
        while index < len(normalized_text):
            character = normalized_text[index]
            current_chars.append(character)
            if character in punctuation:
                lookahead = index + 1
                while lookahead < len(normalized_text) and normalized_text[lookahead] in CLOSING_PUNCTUATION:
                    current_chars.append(normalized_text[lookahead])
                    lookahead += 1
                sentence = "".join(current_chars).strip()
                if sentence:
                    sentences.append(sentence)
                current_chars.clear()
                while lookahead < len(normalized_text) and normalized_text[lookahead].isspace():
                    lookahead += 1
                index = lookahead
                continue
            index += 1
        tail = "".join(current_chars).strip()
        if tail:
            sentences.append(tail)
        return sentences

    @classmethod
    def _join_sentence_parts(cls, left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        if cls._contains_cjk(left) or cls._contains_cjk(right):
            return left + right
        return f"{left} {right}"

    def _split_voice_clone_text(self, text: str, *, max_tokens: int) -> list[str]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []
        safe_max_tokens = max(1, int(max_tokens))
        sentence_candidates = self._split_text_by_punctuation(
            normalized_text,
            SENTENCE_END_PUNCTUATION,
        ) or [normalized_text.strip()]
        sentence_slices: list[tuple[int, str]] = []
        for sentence_text in sentence_candidates:
            normalized_sentence = sentence_text.strip()
            if not normalized_sentence:
                continue
            sentence_token_count = self._count_text_tokens(normalized_sentence)
            if sentence_token_count <= safe_max_tokens:
                sentence_slices.append((sentence_token_count, normalized_sentence))
                continue
            clause_candidates = self._split_text_by_punctuation(normalized_sentence, CLAUSE_SPLIT_PUNCTUATION)
            if len(clause_candidates) <= 1:
                clause_candidates = [normalized_sentence]
            for clause_text in clause_candidates:
                normalized_clause = clause_text.strip()
                if not normalized_clause:
                    continue
                clause_token_count = self._count_text_tokens(normalized_clause)
                if clause_token_count <= safe_max_tokens:
                    sentence_slices.append((clause_token_count, normalized_clause))
                    continue
                for piece in self._split_text_by_token_budget(normalized_clause, safe_max_tokens):
                    normalized_piece = piece.strip()
                    if normalized_piece:
                        sentence_slices.append((self._count_text_tokens(normalized_piece), normalized_piece))
        chunks: list[str] = []
        current_chunk = ""
        current_chunk_token_count = 0
        for sentence_token_count, sentence_text in sentence_slices:
            if not current_chunk:
                current_chunk = sentence_text
                current_chunk_token_count = sentence_token_count
                continue
            if current_chunk_token_count + sentence_token_count > safe_max_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = sentence_text
                current_chunk_token_count = sentence_token_count
            else:
                current_chunk = self._join_sentence_parts(current_chunk, sentence_text)
                current_chunk_token_count = self._count_text_tokens(current_chunk)
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks if len(chunks) > 1 else [normalized_text]

    @staticmethod
    def _estimate_voice_clone_inter_chunk_pause_seconds(text_chunk: str) -> float:
        word_count = len([item for item in str(text_chunk or "").strip().split() if item])
        return (
            DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS
            if word_count <= 4
            else DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS
        )

    def _resolve_prompt_audio_codes(
        self,
        *,
        voice: str | None,
        reference_audio_path: str | Path | None,
    ) -> list[list[int]]:
        if reference_audio_path is not None and str(reference_audio_path).strip():
            return self._encode_reference_audio(str(reference_audio_path))

        normalized_voice = str(voice or "").strip()
        if not normalized_voice:
            available = ", ".join(item["voice"] for item in self.runtime.list_builtin_voices())
            raise ValueError(f"voice is required when reference audio is not provided. Available voices: {available}")

        for row in self.runtime.list_builtin_voices():
            if str(row["voice"]).lower() == normalized_voice.lower():
                return list(row["prompt_audio_codes"])

        available = ", ".join(item["voice"] for item in self.runtime.list_builtin_voices())
        raise ValueError(f"Built-in voice not found: {normalized_voice}. Available voices: {available}")

    def _encode_reference_audio(self, reference_audio_path: str | Path) -> list[list[int]]:
        prepared = prepare_reference_audio(
            reference_audio_path,
            target_sample_rate=self.codec_sample_rate,
            target_channels=self.codec_channels,
        )
        waveform = np.transpose(prepared.waveform, (1, 0))[None, :, :].astype(np.float32, copy=False)
        waveform_length = int(waveform.shape[-1])

        outputs = self.runtime.sessions["codec_encode"].run(
            None,
            {
                "waveform": waveform,
                "input_lengths": np.asarray([waveform_length], dtype=np.int32),
            },
        )
        output_names = [output.name for output in self.runtime.sessions["codec_encode"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_codes = np.asarray(named_outputs["audio_codes"], dtype=np.int32)
        audio_code_lengths = np.asarray(named_outputs["audio_code_lengths"], dtype=np.int32)

        code_length = int(audio_code_lengths.reshape(-1)[0])
        num_quantizers = int(self.runtime.codec_meta["codec_config"]["num_quantizers"])
        prompt_audio_codes: list[list[int]] = []
        for frame_index in range(code_length):
            prompt_audio_codes.append(
                [int(audio_codes[0, frame_index, quantizer_index]) for quantizer_index in range(num_quantizers)]
            )
        return prompt_audio_codes

    @staticmethod
    def _merge_audio_channels(channel_arrays: list[np.ndarray], audio_length: int) -> np.ndarray:
        if not channel_arrays:
            return np.zeros((0, 1), dtype=np.float32)

        length = max(0, int(audio_length))
        trimmed = [np.asarray(channel[:length], dtype=np.float32) for channel in channel_arrays]
        if len(trimmed) == 1:
            return trimmed[0].reshape(-1, 1)
        min_length = min(int(channel.shape[0]) for channel in trimmed)
        if min_length <= 0:
            return np.zeros((0, len(trimmed)), dtype=np.float32)
        return np.stack([channel[:min_length] for channel in trimmed], axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _concat_waveforms(waveforms: list[np.ndarray]) -> np.ndarray:
        if not waveforms:
            return np.zeros((0, 1), dtype=np.float32)
        non_empty = [waveform for waveform in waveforms if waveform.size > 0]
        if not non_empty:
            channel_count = int(waveforms[0].shape[1]) if waveforms[0].ndim == 2 and waveforms[0].shape[1] > 0 else 1
            return np.zeros((0, channel_count), dtype=np.float32)
        return np.concatenate(non_empty, axis=0)
