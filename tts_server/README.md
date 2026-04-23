# tts-server

Torchless FastAPI TTS service with a local ONNX runtime layer under `app/*`.

## What Is In This Repository

- `main.py`: FastAPI entrypoint and route registration
- `app/runtime/*`: local runtime and runtime configuration
- `app/services/*`: TTS service orchestration for batch and stream modes
- `app/utils/*`: torchless audio preprocessing and encoding helpers
- `scripts/*`: acceptance checks and smoke scripts
- `tests/*`: API and runtime verification tests

## Requirements

- Python `>=3.12`
- `uv` package manager (recommended)

## Dependency Matrix

| Scope | Dependency | Required | Notes |
| --- | --- | --- | --- |
| Core runtime | `fastapi`, `uvicorn`, `numpy`, `soundfile`, `soxr` | Yes | Required to start service and run fallback/runtime audio path |
| ONNX inference path | `onnxruntime`, `sentencepiece` | Yes for ONNX mode | Needed when running local MOSS ONNX inference |
| Manual SDK smoke | `openai` | Optional | Only needed for `scripts/smoke_openai_sdk.py` |
| Reference directories | `MOSS-TTS-Nano-main`, `Kokoro-FastAPI-master` | No | Not runtime dependencies; acceptance validates service without them |

## Quick Start

```bash
uv sync
cp .env.example .env
# IMPORTANT: replace API_KEY in .env before startup.
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Alternative startup via `main.py`:

```bash
uv run python main.py
```

Both entrypoints auto-load `.env` from project root. Existing process env vars take precedence over `.env`.

Quick one-off overrides without editing `.env`:

```bash
# Start with auth enabled and an explicit token
API_KEY='local-dev-token-12345' AUTH_REQUIRED=true uv run python main.py

# Temporary migration window (no bearer auth enforcement)
AUTH_REQUIRED=false uv run python main.py

# Keep auth on, but open /docs, /redoc, /openapi.json for local debugging
DOCS_PUBLIC_IN_DEV=true API_KEY='local-dev-token-12345' AUTH_REQUIRED=true uv run python main.py
```

## Docker Quick Start

The repository includes `Dockerfile` and `docker-compose.yml` for container startup.

`models` is mounted from host to container (`./models:/app/models`) so model assets persist across restarts and are not re-downloaded on each `docker compose up`.

```bash
cp .env.example .env
# IMPORTANT: replace API_KEY in .env before startup.

# Put ONNX model files under ./models on host first (or keep existing local files).
docker compose up --build -d
```

Stop service:

```bash
docker compose down
```

If you store models in another host path, edit `docker-compose.yml` volume mapping accordingly.

Service URLs:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`

## Authentication Policy

- `AUTH_REQUIRED=true` (default) protects all HTTP endpoints except `GET /health`.
- Protected endpoints include `/`, `/tts/*`, `/v1/*`, `/docs`, `/redoc`, and `/openapi.json`.
- Unauthorized requests return `HTTP 401` with `WWW-Authenticate: Bearer`.
- Auth token source is only `Authorization: Bearer <API_KEY>`.
- Optional local-dev docs toggle:
  - Set `DOCS_PUBLIC_IN_DEV=true` to exempt `/docs`, `/redoc`, and `/openapi.json` while keeping API routes protected.
  - Keep `DOCS_PUBLIC_IN_DEV=false` in production.
  - Swagger UI exposes an `Authorize` button for bearer token input.
- Rollout toggle:
  - Set `AUTH_REQUIRED=false` for temporary migration windows.
  - Set back to `AUTH_REQUIRED=true` after clients send bearer tokens consistently.

## OpenAI-Compatible Request Examples

Non-stream (`wav`) request:

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" \
  -H "Authorization: Bearer local-dev-token-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello from non-stream API example.",
    "voice": "alloy",
    "response_format": "wav",
    "stream": false
  }' \
  --output /tmp/tts-example.wav
```

Non-stream (`wav`) request with canonical builtin voice (not OpenAI alias):

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" \
  -H "Authorization: Bearer local-dev-token-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello from canonical voice example.",
    "voice": "adam",
    "response_format": "wav",
    "stream": false
  }' \
  --output /tmp/tts-canonical-adam.wav
```

Stream (`pcm`) request:

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" \
  -H "Authorization: Bearer local-dev-token-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello from stream API example.",
    "voice": "alloy",
    "response_format": "pcm",
    "stream": true
  }' \
  --output /tmp/tts-example.pcm
```

Inspect available voices (canonical + alias metadata + prompt-audio status):

```bash
curl -X GET "http://127.0.0.1:8000/v1/audio/voices" \
  -H "Authorization: Bearer local-dev-token-12345"
```

Use custom prompt audio by local server path:

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" \
  -H "Authorization: Bearer local-dev-token-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Use my local prompt audio.",
    "voice": "alloy",
    "response_format": "wav",
    "stream": false,
    "reference_audio_path": "/home/ubuntu/project/public_api_server/tts_server/assets/audio/en_3.wav"
  }' \
  --output /tmp/tts-ref-path.wav
```

Use custom prompt audio by URL:

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" \
  -H "Authorization: Bearer local-dev-token-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Use remote prompt audio.",
    "voice": "alloy",
    "response_format": "wav",
    "stream": false,
    "reference_audio_url": "https://example.com/reference.wav"
  }' \
  --output /tmp/tts-ref-url.wav
```

Use custom prompt audio by file upload (`multipart/form-data`):

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" \
  -H "Authorization: Bearer local-dev-token-12345" \
  -F "model=tts-1" \
  -F "input=Use uploaded prompt audio." \
  -F "voice=alloy" \
  -F "response_format=wav" \
  -F "stream=false" \
  -F "reference_audio_file=@./assets/audio/en_3.wav" \
  --output /tmp/tts-ref-upload.wav
```

## Native `/tts/*` Request Examples

Batch request (`/tts/batch`, returns base64 WAV in unified envelope):

```bash
curl -X POST "http://127.0.0.1:8000/tts/batch" \
  -H "Authorization: Bearer local-dev-token-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from native batch endpoint.",
    "reference_audio_path": "/home/ubuntu/project/public_api_server/tts_server/assets/audio/en_3.wav"
  }'
```

Stream request (`/tts/stream`, returns base64 PCM chunks in unified envelope):

```bash
curl -X POST "http://127.0.0.1:8000/tts/stream" \
  -H "Authorization: Bearer local-dev-token-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from native stream endpoint.",
    "reference_audio_path": "/home/ubuntu/project/public_api_server/tts_server/assets/audio/en_3.wav"
  }'
```

## API Endpoints

All successful `/tts/*` endpoints return a unified envelope:

```json
{
  "code": 0,
  "message": "success",
  "data": {}
}
```

Routes:

- Exempt route:
  - `GET /health` is public and does not require bearer auth.
- Protected route policy:
  - All other routes require `Authorization: Bearer <API_KEY>`.
  - Missing, malformed, or invalid token returns `401` with `WWW-Authenticate: Bearer`.

- `POST /tts/batch`
  - Body: `text`, `reference_audio_path`
  - Returns base64 WAV bytes in `data.audio_base64`
- `POST /tts/stream`
  - Body: `text`, `reference_audio_path`
  - Returns base64 PCM chunks and aggregated bytes
- `GET /v1/models`
  - OpenAI-compatible model discovery
  - Model IDs: `tts-1`, `tts-1-hd`, `moss-tts-nano-onnx`
- `GET /v1/audio/voices`
  - Canonical builtin voices and OpenAI alias mapping
  - Includes prompt-audio availability status
- `POST /v1/audio/speech`
  - OpenAI-compatible speech synthesis endpoint
  - Core fields: `model`, `input`, `voice`, `response_format`, `stream`
  - Supported `response_format`: `wav`, `pcm`
  - Optional reference-audio fields (higher priority than builtin `voice`):
    - `reference_audio_file` (multipart file)
    - `reference_audio_url` (http/https)
    - `reference_audio_path` (local path)
  - Reference-audio priority order:
    - `reference_audio_file` > `reference_audio_url` > `reference_audio_path` > builtin `voice`
  - `voice` compatibility:
    - OpenAI-style aliases are configured in `app/config/openai_mappings.json` (default 6 aliases: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`)
    - Canonical builtin voice names are also accepted (default runtime fallback voice set has 16 names)
    - Canonical voice -> prompt audio filename mapping is defined in `app/runtime/onnx_runtime.py`
  - Builtin prompt audio usage:
    - Used when no custom `reference_audio_*` is provided
    - Runtime resolves prompt files from `TTS_PROMPT_AUDIO_DIR` or default `./assets/audio`
  - Error semantics:
    - 4xx for invalid client input (`model`, `input`, `response_format`, invalid reference source)
    - 5xx for runtime/inference failures

## Concurrency And Async Notes

- Client side: all endpoints can be called from async clients (such as `httpx.AsyncClient`) or normal blocking clients.
- Server side route declaration:
  - `/v1/audio/speech` is declared as an async route.
  - `/tts/batch` and `/tts/stream` are declared as sync routes.
- Important behavior:
  - Route declaration style does not make TTS inference itself non-blocking; synthesis is still compute/runtime work.
  - `/v1/audio/speech` with `"stream": true` returns `StreamingResponse` (chunked HTTP body).
  - `/tts/stream` returns all chunks aggregated in JSON (not chunked HTTP streaming).

## Frontend Integration Guide

Use this quick decision guide when choosing between endpoints:

| Scenario | Recommended Endpoint | Why |
| --- | --- | --- |
| OpenAI SDK compatibility / vendor-like API shape | `POST /v1/audio/speech` | Request/response fields align with OpenAI-style TTS semantics. |
| Browser playback that should start as early as possible | `POST /v1/audio/speech` + `"stream": true` | Returns chunked HTTP audio stream for progressive consumption. |
| One-shot download or save to file (`wav`/`pcm`) | `POST /v1/audio/speech` + `"stream": false` | Returns raw audio bytes directly. |
| Internal debugging with envelope-style JSON payloads | `POST /tts/batch` or `POST /tts/stream` | Unified `{code,message,data}` shape is easy to inspect/log. |
| Need chunk list as JSON (not HTTP chunk stream) | `POST /tts/stream` | Returns base64 chunk list plus aggregated bytes in one JSON response. |

Practical recommendations:

- For production frontend players, prefer `/v1/audio/speech`:
  - Real-time-ish playback: `stream=true`.
  - Full-file playback/download: `stream=false`.
- Keep `/tts/batch` and `/tts/stream` for internal tools, diagnostics, and compatibility checks with the unified response envelope.
- If you pass custom reference audio (`reference_audio_file` / `reference_audio_url` / `reference_audio_path`), the server prioritizes that over builtin voice prompt assets.
- If you do not pass custom reference audio, voice resolution falls back to builtin mappings (OpenAI-style alias -> canonical voice -> prompt file in `assets/audio` or `TTS_PROMPT_AUDIO_DIR`).

## Environment Variables

- `APP_HOST`: server host, default `0.0.0.0`
- `APP_PORT`: server port, default `8000`
- `AUTH_REQUIRED`: bearer-auth switch, default `1` (`true`)
- `DOCS_PUBLIC_IN_DEV`: docs exemption switch, default `0` (`false`)
  - `1` (`true`) exempts `/docs`, `/redoc`, `/openapi.json` from bearer auth (intended for local dev only)
- `API_KEY`: bearer token secret used for protected routes
  - Must be non-empty and not placeholder-like when `AUTH_REQUIRED=true`
  - Startup fails fast if invalid in required-auth mode
- `ONNX_MODEL_DIR`: optional ONNX model directory
  - If unset, runtime auto-uses local `./models` when present
- `TTS_PROMPT_AUDIO_DIR`: optional builtin voice prompt-audio directory override
  - If unset, runtime auto-uses local `./assets/audio` when present
- `TTS_SAMPLE_RATE`: output sample rate, default `16000`
- `TTS_CHANNELS`: output channels, default `1`
- `TTS_STREAM_CHUNK_SAMPLES`: stream chunk size in samples, default `3200`
- `ONNX_CPU_THREADS`: ONNX runtime thread count, default `4`
- `ONNX_MAX_NEW_FRAMES`: generation frame budget (recommended `375`)
- `ONNX_SAMPLE_MODE`: `fixed` / `full` / `greedy` (recommended `fixed`)
- `ONNX_DO_SAMPLE`: sampling toggle (`1` or `0`, recommended `1`)
- `ONNX_TEXT_TEMPERATURE`: text-token temperature (recommended `1.0`)
- `ONNX_TEXT_TOP_P`: text-token top-p (recommended `1.0`)
- `ONNX_TEXT_TOP_K`: text-token top-k (recommended `50`)
- `ONNX_AUDIO_TEMPERATURE`: audio-token temperature (recommended `0.8`)
- `ONNX_AUDIO_TOP_P`: audio-token top-p (recommended `0.95`)
- `ONNX_AUDIO_TOP_K`: audio-token top-k (recommended `25`)
- `ONNX_AUDIO_REPETITION_PENALTY`: audio repetition penalty (recommended `1.2`)
- `ONNX_SEED`: request-level RNG seed; default `0`
- `ONNX_VOICE_CLONE_MAX_TEXT_TOKENS`: sentence chunk token budget (recommended `75`)
- `ONNX_ENABLE_NORMALIZE_TTS_TEXT`: text pre-normalization switch (`1` by default)

## Acceptance Workflow

Automated acceptance workflow:

```bash
./scripts/run_onnx_api_acceptance.sh
```

The workflow runs:

1. `tests/test_onnx_api_acceptance.py` (stream, batch, invalid model, empty input)
2. `tests/test_torchless_startup.py` (service usability without `torch`/`torchaudio`)
3. `scripts/smoke_without_reference_dirs.py` (reference-dir independence checks with scenario-level PASS/FAIL output)

Manual smoke commands:

```bash
./scripts/smoke_openai_curl.sh
uv run python scripts/smoke_openai_sdk.py
```

You can override smoke parameters with env vars such as `OPENAI_BASE_URL`, `OPENAI_API_KEY` (or `API_KEY`), `SMOKE_MODEL`, `SMOKE_TEXT`, `SMOKE_RESPONSE_FORMAT`, and `SMOKE_OUTPUT_FILE`.

## Release Precheck (Required)

Before delivery, all checks below must pass in order:

1. `uv run python scripts/check_runtime_import_boundaries.py`
2. `./scripts/run_onnx_api_acceptance.sh`
3. `uv run python scripts/smoke_without_reference_dirs.py`

The precheck is only valid if the service remains callable when `MOSS-TTS-Nano-main` and `Kokoro-FastAPI-master` are absent (the smoke script enforces this condition by temporarily renaming both directories).

## Known Limitations And Non-Goals

- `POST /v1/audio/speech` only supports `response_format=wav|pcm`.
- This project does not include comprehensive load/performance benchmarking in acceptance.
- This project does not include subjective multi-language voice-quality evaluation.

## Local Prompt Assets

The repository includes local `assets/` data (audio/images/videos/demo metadata). Runtime builtin-voice prompt resolution reads `assets/audio` by default, so `/v1` synthesis is independent from external reference repositories.

## ONNX Runtime Mode

When `onnxruntime` is available and `ONNX_MODEL_DIR` resolves to a valid model bundle, runtime uses local MOSS ONNX inference for `/tts/*` and `/v1/audio/speech`. The sinusoid fallback path is only used when ONNX runtime/model initialization is unavailable.

## Presets

Ready-to-use runtime presets:

- `.env.preset.natural`: prioritize expressive/prosodic output
- `.env.preset.stable`: prioritize conservative/stable output

Switch preset:

```bash
./scripts/apply_env_preset.sh natural
# or
./scripts/apply_env_preset.sh stable
```

After applying a preset, replace `API_KEY` in `.env` with a real secret token before startup.

Then restart the service:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```
