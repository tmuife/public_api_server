from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LABEL_PATTERN = re.compile(r"^\[(.+?)\]\s*$")
ALLOWED_CONFIG_KEYS = {
    "model",
    "voice",
    "response_format",
    "stream",
    "reference_audio_path",
    "reference_audio_url",
}
KNOWN_LABEL_SLUGS = {
    "慢速": "slow",
    "正常": "normal",
    "中文": "zh",
    "讲解": "explain",
    "强化": "reinforce",
}


@dataclass(frozen=True)
class Segment:
    index: int
    word: str
    label: str
    text: str
    source_line: int


@dataclass(frozen=True)
class Defaults:
    model: str
    voice: str
    response_format: str


def _load_project_dotenv() -> None:
    """Load project-root .env into os.environ without overriding existing keys."""
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key_part, value_part = line.split("=", 1)
        key = key_part.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key or key in os.environ:
            continue

        value = value_part.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _read_env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value if value else default


def _resolve_api_key() -> str:
    for name in ("TTS_API_KEY", "API_KEY", "OPENAI_API_KEY"):
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


def _slugify_ascii(value: str) -> str:
    known = KNOWN_LABEL_SLUGS.get(value.strip())
    if known:
        return known
    lowered = value.lower()
    safe = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return safe or "segment"


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"cannot coerce bool from value={value!r}")


def _normalize_response_format(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in {"wav", "pcm"}:
        raise ValueError(f"response_format must be wav or pcm, got: {value!r}")
    return normalized


def _extract_word_marker(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None

    if stripped.startswith("#"):
        heading = stripped.lstrip("#").strip()
        return heading or None

    lowered = stripped.lower()
    for prefix in ("word:", "word：", "单词:", "单词："):
        if lowered.startswith(prefix):
            value = stripped[len(prefix) :].strip()
            return value or None
    return None


def parse_podcast_markdown(path: Path) -> list[Segment]:
    lines = path.read_text(encoding="utf-8").splitlines()

    segments: list[Segment] = []
    current_label: str | None = None
    current_text_lines: list[str] = []
    current_source_line = -1
    current_word = path.stem
    segment_word = path.stem

    def flush_segment() -> None:
        nonlocal current_label, current_text_lines, current_source_line, segment_word
        if current_label is None:
            return

        trimmed_lines = current_text_lines[:]
        while trimmed_lines and not trimmed_lines[0].strip():
            trimmed_lines.pop(0)
        while trimmed_lines and not trimmed_lines[-1].strip():
            trimmed_lines.pop()

        text = "\n".join(trimmed_lines).strip()
        if text:
            segments.append(
                Segment(
                    index=len(segments) + 1,
                    word=segment_word,
                    label=current_label,
                    text=text,
                    source_line=current_source_line,
                )
            )
        else:
            print(
                f"[WARN] Skip empty segment label=[{current_label}] at line {current_source_line}",
                file=sys.stderr,
            )

        current_label = None
        current_text_lines = []
        current_source_line = -1
        segment_word = current_word

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n")
        if current_label is None:
            marker_word = _extract_word_marker(line)
            if marker_word:
                current_word = marker_word
                segment_word = marker_word
                continue

        match = LABEL_PATTERN.match(line.strip())
        if match is not None:
            flush_segment()
            label = match.group(1).strip()
            current_label = label
            current_source_line = line_number
            segment_word = current_word
            continue

        if current_label is None:
            continue

        current_text_lines.append(line)

    flush_segment()
    return segments


def load_style_config(path: Path | None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if path is None:
        return {}, {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("style config root must be an object")

    default_overrides_raw = payload.get("default", {})
    labels_raw = payload.get("labels", {})

    if not isinstance(default_overrides_raw, dict):
        raise ValueError("style config key 'default' must be an object")
    if not isinstance(labels_raw, dict):
        raise ValueError("style config key 'labels' must be an object")

    default_overrides = _normalize_style_overrides(default_overrides_raw, context="default")

    labels: dict[str, dict[str, Any]] = {}
    for label, overrides_raw in labels_raw.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label keys in style config must be non-empty strings")
        if not isinstance(overrides_raw, dict):
            raise ValueError(f"style config labels['{label}'] must be an object")
        labels[label.strip()] = _normalize_style_overrides(overrides_raw, context=f"label={label!r}")

    return default_overrides, labels


def _normalize_style_overrides(raw: dict[str, Any], context: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in ALLOWED_CONFIG_KEYS:
            raise ValueError(
                f"unsupported key '{key}' in style config ({context}); "
                f"allowed keys: {sorted(ALLOWED_CONFIG_KEYS)}"
            )

        if key == "response_format":
            normalized[key] = _normalize_response_format(str(value))
            continue

        if key == "stream":
            normalized[key] = _coerce_bool(value)
            continue

        if key in {"model", "voice", "reference_audio_path", "reference_audio_url"}:
            text_value = str(value).strip()
            if not text_value:
                continue
            normalized[key] = text_value
            continue

    return normalized


def build_payload(
    segment: Segment,
    defaults: Defaults,
    default_overrides: dict[str, Any],
    label_overrides: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": defaults.model,
        "input": segment.text,
        "voice": defaults.voice,
        "response_format": defaults.response_format,
        "stream": False,
    }

    payload.update(default_overrides)
    payload.update(label_overrides.get(segment.label, {}))

    payload["response_format"] = _normalize_response_format(str(payload.get("response_format", defaults.response_format)))
    payload["model"] = str(payload.get("model", defaults.model)).strip() or defaults.model
    payload["voice"] = str(payload.get("voice", defaults.voice)).strip() or defaults.voice
    payload["stream"] = _coerce_bool(payload.get("stream", False))

    for optional_key in ("reference_audio_path", "reference_audio_url"):
        if optional_key in payload:
            text_value = str(payload.get(optional_key, "")).strip()
            if text_value:
                payload[optional_key] = text_value
            else:
                payload.pop(optional_key, None)

    return payload


def invoke_tts_api(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: float,
) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc


def concat_audio_files(segment_paths: list[Path], output_file: Path, response_format: str) -> int:
    normalized_format = _normalize_response_format(response_format)
    if not segment_paths:
        raise ValueError("segment_paths cannot be empty")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if normalized_format == "pcm":
        total_bytes = 0
        with output_file.open("wb") as writer:
            for segment_path in segment_paths:
                chunk = segment_path.read_bytes()
                writer.write(chunk)
                total_bytes += len(chunk)
        return total_bytes

    if normalized_format != "wav":
        raise ValueError(f"unsupported concat format: {normalized_format}")

    wave_params: tuple[int, int, int, str, str] | None = None
    frame_chunks: list[bytes] = []
    total_bytes = 0

    for segment_path in segment_paths:
        with wave.open(str(segment_path), "rb") as reader:
            params = (
                reader.getnchannels(),
                reader.getsampwidth(),
                reader.getframerate(),
                reader.getcomptype(),
                reader.getcompname(),
            )
            if wave_params is None:
                wave_params = params
            elif params != wave_params:
                raise ValueError(
                    "wav params mismatch while concatenating: "
                    f"expected={wave_params}, actual={params}, file={segment_path}"
                )
            frames = reader.readframes(reader.getnframes())
            frame_chunks.append(frames)
            total_bytes += len(frames)

    if wave_params is None:
        raise ValueError("unable to read wav params from segments")

    with wave.open(str(output_file), "wb") as writer:
        writer.setnchannels(wave_params[0])
        writer.setsampwidth(wave_params[1])
        writer.setframerate(wave_params[2])
        writer.setcomptype(wave_params[3], wave_params[4])
        writer.writeframes(b"".join(frame_chunks))

    return total_bytes


def _build_concat_file_name(input_stem: str, word: str, response_format: str, group_count: int) -> str:
    word_slug = _slugify_ascii(word)
    if group_count == 1:
        return f"{input_stem}_full.{response_format}"
    return f"{input_stem}_{word_slug}_full.{response_format}"


def run_concat(
    *,
    manifest_rows: list[dict[str, Any]],
    output_dir: Path,
    input_stem: str,
    overwrite: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    eligible_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        if row.get("status") not in {"success", "skipped_exists", "dry_run"}:
            continue
        if not row.get("output_file"):
            continue
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            continue
        response_format = str(payload.get("response_format", "")).strip().lower()
        if response_format not in {"wav", "pcm"}:
            continue
        eligible_rows.append(row)

    group_keys = sorted(
        {
            (
                str(row.get("word", input_stem)).strip() or input_stem,
                str(row.get("payload", {}).get("response_format", "wav")).strip().lower(),
            )
            for row in eligible_rows
        },
        key=lambda item: item[0],
    )
    group_count = len(group_keys)

    concat_rows: list[dict[str, Any]] = []
    for word, response_format in group_keys:
        segment_rows = [
            row
            for row in eligible_rows
            if (str(row.get("word", input_stem)).strip() or input_stem) == word
            and str(row.get("payload", {}).get("response_format", "")).strip().lower() == response_format
        ]
        segment_rows.sort(key=lambda item: int(item.get("index", 0)))
        segment_paths = [Path(str(item["output_file"])) for item in segment_rows]

        concat_file_name = _build_concat_file_name(
            input_stem=input_stem,
            word=word,
            response_format=response_format,
            group_count=group_count,
        )
        concat_path = output_dir / concat_file_name

        concat_row: dict[str, Any] = {
            "word": word,
            "response_format": response_format,
            "segment_count": len(segment_paths),
            "segment_indices": [int(item.get("index", 0)) for item in segment_rows],
            "segment_files": [str(path) for path in segment_paths],
            "output_file": str(concat_path),
            "status": "pending",
        }

        if dry_run:
            concat_row["status"] = "dry_run"
            concat_rows.append(concat_row)
            print(
                f"[DRY ] [concat] word={word} format={response_format} "
                f"segments={len(segment_paths)} -> {concat_path.name}"
            )
            continue

        if concat_path.exists() and not overwrite:
            concat_row["status"] = "skipped_exists"
            concat_rows.append(concat_row)
            print(f"[SKIP] [concat] file exists: {concat_path}")
            continue

        missing_paths = [path for path in segment_paths if not path.exists()]
        if missing_paths:
            concat_row["status"] = "failed"
            concat_row["error"] = f"missing segment files: {[str(path) for path in missing_paths]}"
            concat_rows.append(concat_row)
            print(f"[FAIL] [concat] missing segment files for word={word}", file=sys.stderr)
            continue

        try:
            byte_length = concat_audio_files(
                segment_paths=segment_paths,
                output_file=concat_path,
                response_format=response_format,
            )
            concat_row["status"] = "success"
            concat_row["byte_length"] = byte_length
            concat_rows.append(concat_row)
            print(
                f"[PASS] [concat] word={word} format={response_format} "
                f"segments={len(segment_paths)} bytes={byte_length}"
            )
        except Exception as exc:
            concat_row["status"] = "failed"
            concat_row["error"] = str(exc)
            concat_rows.append(concat_row)
            print(f"[FAIL] [concat] word={word} {exc}", file=sys.stderr)

    return concat_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse docs/podcast.md and call /v1/audio/speech to generate audio per bracket segment."
    )
    parser.add_argument("--input", default="docs/podcast.md", help="Path to podcast markdown file.")
    parser.add_argument(
        "--output-dir",
        default="outputs/podcast_audio",
        help="Directory to save generated audio and manifest.",
    )
    parser.add_argument(
        "--base-url",
        default=_read_env("TTS_BASE_URL", "http://127.0.0.1:8000"),
        help="TTS server base URL, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--endpoint",
        default=_read_env("TTS_SPEECH_ENDPOINT", "/v1/audio/speech"),
        help="Speech endpoint path. Default: /v1/audio/speech",
    )
    parser.add_argument(
        "--api-key",
        default=_resolve_api_key(),
        help="Bearer token. Default resolves from TTS_API_KEY/API_KEY/OPENAI_API_KEY",
    )
    parser.add_argument("--default-model", default=_read_env("TTS_MODEL", "tts-1"), help="Default model.")
    parser.add_argument("--default-voice", default=_read_env("TTS_DEFAULT_VOICE", "alloy"), help="Default voice.")
    parser.add_argument(
        "--default-response-format",
        default=_read_env("TTS_RESPONSE_FORMAT", "wav"),
        help="Default audio format: wav|pcm",
    )
    parser.add_argument(
        "--style-config",
        default="",
        help="Optional JSON file to map [label] -> API overrides.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(_read_env("TTS_TIMEOUT", "120")),
        help="HTTP timeout in seconds.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and build payload only, without API requests.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed segment.")
    parser.add_argument("--concat", action="store_true", help="Concatenate generated segment audio into per-word files.")
    return parser.parse_args()


def main() -> int:
    _load_project_dotenv()
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    style_config_path = Path(args.style_config).expanduser().resolve() if args.style_config else None

    try:
        default_response_format = _normalize_response_format(args.default_response_format)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    defaults = Defaults(
        model=str(args.default_model).strip() or "tts-1",
        voice=str(args.default_voice).strip() or "alloy",
        response_format=default_response_format,
    )

    if not input_path.exists():
        print(f"[FAIL] Input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        segments = parse_podcast_markdown(input_path)
    except Exception as exc:
        print(f"[FAIL] Parse markdown failed: {exc}", file=sys.stderr)
        return 2

    if not segments:
        print("[FAIL] No segments found. Ensure markdown contains lines like [标签] followed by text.", file=sys.stderr)
        return 2

    try:
        default_overrides, label_overrides = load_style_config(style_config_path)
    except Exception as exc:
        print(f"[FAIL] Load style config failed: {exc}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint_path = str(args.endpoint).strip()
    if not endpoint_path.startswith("/"):
        endpoint_path = f"/{endpoint_path}"
    request_url = f"{str(args.base_url).rstrip('/')}{endpoint_path}"

    manifest_rows: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Segments detected: {len(segments)}")
    print(f"[INFO] Endpoint: {request_url}")
    print(f"[INFO] Output directory: {output_dir}")
    if style_config_path is not None:
        print(f"[INFO] Style config: {style_config_path}")

    for segment in segments:
        payload = build_payload(segment, defaults, default_overrides, label_overrides)

        audio_ext = str(payload["response_format"])
        label_slug = _slugify_ascii(segment.label)
        file_name = f"{input_path.stem}_{segment.index:03d}_{label_slug}.{audio_ext}"
        output_file = output_dir / file_name

        row: dict[str, Any] = {
            "index": segment.index,
            "word": segment.word,
            "label": segment.label,
            "source_line": segment.source_line,
            "text": segment.text,
            "payload": payload,
            "output_file": str(output_file),
            "status": "pending",
        }

        if output_file.exists() and not args.overwrite and not args.dry_run:
            row["status"] = "skipped_exists"
            manifest_rows.append(row)
            print(f"[SKIP] #{segment.index:03d} [{segment.label}] file exists: {output_file}")
            continue

        if args.dry_run:
            row["status"] = "dry_run"
            manifest_rows.append(row)
            print(f"[DRY ] #{segment.index:03d} [{segment.label}] {output_file.name}")
            success_count += 1
            continue

        print(f"[RUN ] #{segment.index:03d} [{segment.label}] -> {output_file.name}")
        try:
            audio_bytes = invoke_tts_api(
                url=request_url,
                payload=payload,
                api_key=str(args.api_key).strip(),
                timeout_seconds=float(args.timeout),
            )
            if len(audio_bytes) <= 0:
                raise RuntimeError("empty audio response")

            output_file.write_bytes(audio_bytes)
            row["status"] = "success"
            row["byte_length"] = len(audio_bytes)
            success_count += 1
            print(f"[PASS] #{segment.index:03d} bytes={len(audio_bytes)}")
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            failure_count += 1
            print(f"[FAIL] #{segment.index:03d} [{segment.label}] {exc}", file=sys.stderr)
            if args.fail_fast:
                manifest_rows.append(row)
                break

        manifest_rows.append(row)

    concat_rows: list[dict[str, Any]] = []
    if args.concat:
        concat_rows = run_concat(
            manifest_rows=manifest_rows,
            output_dir=output_dir,
            input_stem=input_path.stem,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )

    manifest_path = output_dir / f"{input_path.stem}_manifest.json"
    concat_success_count = sum(1 for row in concat_rows if row.get("status") == "success")
    concat_failure_count = sum(1 for row in concat_rows if row.get("status") == "failed")
    manifest = {
        "input_file": str(input_path),
        "request_url": request_url,
        "total_segments": len(segments),
        "success_count": success_count,
        "failure_count": failure_count,
        "rows": manifest_rows,
        "concat_enabled": bool(args.concat),
        "concat_success_count": concat_success_count,
        "concat_failure_count": concat_failure_count,
        "concat_rows": concat_rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] Manifest written: {manifest_path}")
    print(f"[INFO] Summary: success={success_count}, failed={failure_count}, total={len(segments)}")
    if args.concat:
        print(
            f"[INFO] Concat summary: success={concat_success_count}, "
            f"failed={concat_failure_count}, total={len(concat_rows)}"
        )

    return 0 if (failure_count == 0 and concat_failure_count == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
