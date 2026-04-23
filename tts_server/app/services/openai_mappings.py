from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ALIASES: dict[str, str] = {
    "tts-1": "moss-tts-nano-onnx",
    "tts-1-hd": "moss-tts-nano-onnx",
    "moss-tts-nano": "moss-tts-nano-onnx",
    "moss-tts-nano-onnx": "moss-tts-nano-onnx",
}

DEFAULT_VOICE_ALIASES: dict[str, str] = {
    "alloy": "Adam",
    "echo": "Bella",
    "fable": "Yuewen",
    "onyx": "Adam",
    "nova": "Ava",
    "shimmer": "Lingyu",
}

DEFAULT_MAPPINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "openai_mappings.json"


@dataclass(frozen=True)
class OpenAiMappings:
    public_model_ids: tuple[str, ...]
    model_alias_lookup: dict[str, str]
    public_voice_aliases: tuple[str, ...]
    voice_alias_lookup: dict[str, str]


def load_openai_mappings(mapping_path: Path | None = None) -> OpenAiMappings:
    path = mapping_path or DEFAULT_MAPPINGS_PATH
    models = dict(DEFAULT_MODEL_ALIASES)
    voices = dict(DEFAULT_VOICE_ALIASES)

    if path.exists():
        payload = _read_mapping_payload(path)
        models.update(_as_clean_mapping(payload.get("models")))
        voices.update(_as_clean_mapping(payload.get("voices")))

    model_alias_lookup: dict[str, str] = {}
    for public_model_id, canonical_model_id in models.items():
        model_alias_lookup[public_model_id.lower()] = canonical_model_id
        model_alias_lookup.setdefault(canonical_model_id.lower(), canonical_model_id)

    voice_alias_lookup: dict[str, str] = {}
    for voice_alias, canonical_voice in voices.items():
        voice_alias_lookup[voice_alias.lower()] = canonical_voice

    public_model_ids = tuple(dict.fromkeys(models.keys()))
    public_voice_aliases = tuple(dict.fromkeys(voices.keys()))

    return OpenAiMappings(
        public_model_ids=public_model_ids,
        model_alias_lookup=model_alias_lookup,
        public_voice_aliases=public_voice_aliases,
        voice_alias_lookup=voice_alias_lookup,
    )


def _read_mapping_payload(path: Path) -> dict[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _as_clean_mapping(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}

    cleaned: dict[str, str] = {}
    for key, value in payload.items():
        alias = str(key or "").strip()
        canonical = str(value or "").strip()
        if not alias or not canonical:
            continue
        cleaned[alias] = canonical
    return cleaned
