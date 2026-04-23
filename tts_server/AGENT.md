# AGENT.md

## Purpose

This directory hosts a FastAPI-based torchless TTS API.
Runtime code is locally owned in `app/*`, and reference directories are treated as non-runtime assets.

## Current capabilities

- Start a FastAPI app from `main.py`
- Return unified success responses with `code/message/data`
- Expose endpoints:
  - `GET /`: service metadata
  - `GET /health`: health status
  - `POST /tts/batch`: non-stream synthesis (base64 WAV bytes)
  - `POST /tts/stream`: stream-mode synthesis (base64 PCM chunks + aggregated bytes)
  - `GET /v1/models`: OpenAI-compatible model discovery
  - `GET /v1/audio/voices`: voice alias/canonical mapping visibility
  - `POST /v1/audio/speech`: OpenAI-compatible speech synthesis (`stream` + non-stream)
- Preprocess reference audio without `torch/torchaudio`
  - `soundfile` for I/O
  - `soxr` for resampling
- Resolve voice sources with deterministic priority
  - builtin voice (supports both OpenAI aliases and MOSS canonical names when no reference audio is supplied)
  - reference audio via upload/url/path (overrides builtin voice when provided)
- Restrict OpenAI-compatible `response_format` to `wav` and `pcm`, returning 4xx on unsupported values
- Distinguish error classes for OpenAI-compatible endpoints
  - 4xx: client parameter/source errors
  - 5xx: synthesis/runtime failures

## How to run

```bash
uv sync
cp .env.example .env
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## Core runtime parameters

- `APP_HOST`: bind host, default `0.0.0.0`
- `APP_PORT`: bind port, default `8000`
- `API_KEY`: reserved for endpoint authentication in future versions
- `ONNX_MODEL_DIR`: optional path to local ONNX assets
- `TTS_PROMPT_AUDIO_DIR`: optional override path to prompt-speech assets for canonical MOSS voices
  - If not set, runtime auto-uses local `assets/audio` when present
- `TTS_SAMPLE_RATE`: target sample rate, default `16000`
- `TTS_CHANNELS`: output channels, default `1`
- `TTS_STREAM_CHUNK_SAMPLES`: stream chunk size, default `3200`

## Design notes

- `main.py` only handles routing and response schema.
- Runtime logic lives in `app/runtime`; service orchestration lives in `app/services`.
- Runtime import boundaries are enforced by `scripts/check_runtime_import_boundaries.py`.
- `MOSS-TTS-Nano-main` and `Kokoro-FastAPI-master` must not be imported by runtime path.
- Local prompt assets live under `assets/audio` to keep canonical voice synthesis independent from reference directories.
