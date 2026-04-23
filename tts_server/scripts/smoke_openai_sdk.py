from __future__ import annotations

import os
from pathlib import Path


def _read_env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value if value else default


def _read_required_token() -> str:
    token = str(os.getenv("OPENAI_API_KEY", os.getenv("API_KEY", ""))).strip()
    if token:
        return token
    raise RuntimeError(
        "Missing OPENAI_API_KEY (or API_KEY). "
        "Set it to match server API_KEY before running smoke."
    )


def _extract_audio_bytes(response: object) -> bytes:
    if isinstance(response, (bytes, bytearray)):
        return bytes(response)

    read_method = getattr(response, "read", None)
    if callable(read_method):
        payload = read_method()
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)

    content = getattr(response, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)

    raise RuntimeError("Unable to decode SDK response bytes from client.audio.speech.create(...)")


def main() -> int:
    """
    OpenAI SDK smoke script.

    Prerequisites:
    - Service is running locally (default: http://127.0.0.1:8000/v1)
    - `openai` package installed (`uv add openai`)

    Environment variables:
    - OPENAI_BASE_URL: API base URL (default: http://127.0.0.1:8000/v1)
    - OPENAI_API_KEY: API key / bearer token (required; fallback reads API_KEY)
    - SMOKE_MODEL: model id (default: tts-1)
    - SMOKE_VOICE: voice name (default: alloy)
    - SMOKE_TEXT: synthesis text
    - SMOKE_RESPONSE_FORMAT: wav or pcm (default: wav)
    - SMOKE_STREAM: true or false (default: false)
    - SMOKE_OUTPUT_FILE: output audio path
    """

    try:
        from openai import OpenAI
    except ImportError:
        print("[FAIL] Missing dependency: openai. Install it with `uv add openai`.")
        return 2

    base_url = _read_env("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    try:
        api_key = _read_required_token()
    except RuntimeError as exc:
        print(f"[FAIL] {exc}")
        return 1
    model = _read_env("SMOKE_MODEL", "tts-1")
    voice = _read_env("SMOKE_VOICE", "alloy")
    text = _read_env("SMOKE_TEXT", "OpenAI SDK smoke request from tts-server.")
    response_format = _read_env("SMOKE_RESPONSE_FORMAT", "wav").lower()
    stream_value = _read_env("SMOKE_STREAM", "false").lower()
    output_file = _read_env("SMOKE_OUTPUT_FILE", f"/tmp/tts-smoke-sdk.{response_format}")

    if response_format not in {"wav", "pcm"}:
        print(f"[FAIL] SMOKE_RESPONSE_FORMAT must be wav or pcm, got: {response_format}")
        return 1

    if stream_value not in {"true", "false"}:
        print(f"[FAIL] SMOKE_STREAM must be true or false, got: {stream_value}")
        return 1

    stream = stream_value == "true"

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
            stream=stream,
        )
    except Exception as exc:
        print(f"[FAIL] SDK request failed: {exc}")
        return 1

    try:
        audio_bytes = _extract_audio_bytes(response)
    except Exception as exc:
        print(f"[FAIL] SDK response parse failed: {exc}")
        return 1

    if len(audio_bytes) <= 0:
        print("[FAIL] SDK smoke returned empty audio payload")
        return 1

    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(audio_bytes)

    print(f"[PASS] SDK smoke succeeded ({len(audio_bytes)} bytes). Output: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
