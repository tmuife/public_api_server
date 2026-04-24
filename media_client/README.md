# media_client

`media_client` is a FastAPI service for end-to-end video processing:

1. preprocess a source video (path or upload),
2. extract frames + audio,
3. encrypt each frame with AES-256-GCM,
4. publish encrypted frames to Kafka,
5. compose output video from a result topic.

## Features

- ASGI entrypoint: `main:app`
- Minimal business APIs:
  - `POST /videos/process-by-path`
  - `POST /videos/process-upload`
  - `POST /videos/compose`
- Unified success envelope:
  - `{ "code": 0, "message": "success", "data": ... }`
- `WORK_DIR`-based per-job workspace lifecycle
- Dynamic Kafka input topic bootstrap (`{job_name}_input`)
- AES-256-GCM frame encryption contract

## Requirements

- Python 3.12+
- FFmpeg + FFprobe available in PATH
- Kafka cluster reachable with configured security settings

## Quick Start

```bash
cd media_client
uv sync
cp .env.example .env
```

Configure encryption key for `.env` (choose one option):

Option A (recommended): passphrase + salt (key derived with PBKDF2-HMAC-SHA256):

```bash
python - <<'PY'
import base64, os
print("AES_256_GCM_PASSPHRASE=your-memorable-passphrase")
print("AES_256_GCM_KDF_SALT_BASE64=" + base64.b64encode(os.urandom(16)).decode())
print("AES_256_GCM_KDF_ITERATIONS=600000")
PY
```

Option B (legacy): raw Base64 key:

```bash
python - <<'PY'
import base64, os
print(base64.b64encode(os.urandom(32)).decode())
PY
```

Start service:

```bash
uv run python main.py
```

Or with ASGI style:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Open docs:

```bash
open http://127.0.0.1:8000/docs
```

## Environment Variables

Required core settings:

- `MEDIA_API_ACCESS_TOKEN` (Bearer token used by all APIs except `/health`, `/docs`, `/openapi.json`)
- `WORK_DIR`
- one of:
  - `AES_256_GCM_PASSPHRASE` + `AES_256_GCM_KDF_SALT_BASE64` (derived key mode, recommended)
  - `AES_256_GCM_KEY_BASE64` (must decode to exactly 32 bytes, legacy mode)
- `KAFKA_BOOTSTRAP_SERVERS`

Optional/commonly used settings:

- `UPLOAD_DIR` (default: `${WORK_DIR}/uploads`)
- `KAFKA_SECURITY_PROTOCOL` (`SASL_SSL` default)
- `KAFKA_SASL_MECHANISM`
- `KAFKA_SASL_USERNAME`
- `KAFKA_SASL_PASSWORD`
- `KAFKA_SSL_CA_FILE`
- `KAFKA_PRODUCE_TIMEOUT_SECONDS`
- `KAFKA_CONSUME_TIMEOUT_SECONDS`
- `KAFKA_TOPIC_PARTITIONS`
- `KAFKA_TOPIC_REPLICATION_FACTOR`
- `AES_256_GCM_KDF_ITERATIONS` (default: `600000`, only for passphrase mode)

## API Examples

### 1) Preprocess by local path

```bash
curl -X POST "http://127.0.0.1:8000/videos/process-by-path" \
  -H "Authorization: Bearer ${MEDIA_API_ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/absolute/path/to/source.mp4"}'
```

Example success:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "job_name": "job_1776945123456_ab12cd34",
    "input_topic": "job_1776945123456_ab12cd34_input",
    "manifest_summary": {
      "frame_count": 240,
      "fps": 24.0,
      "bitrate": 1800000,
      "frame_format": "jpg",
      "audio_path": "/abs/workdir/jobs/job_.../audio/source_audio.m4a"
    }
  }
}
```

### 2) Preprocess by upload

```bash
curl -X POST "http://127.0.0.1:8000/videos/process-upload" \
  -H "Authorization: Bearer ${MEDIA_API_ACCESS_TOKEN}" \
  -F "file=@/absolute/path/to/source.mp4"
```

Returns the same response schema as `process-by-path`.

### 3) Compose from full topic

```bash
curl -X POST "http://127.0.0.1:8000/videos/compose" \
  -H "Authorization: Bearer ${MEDIA_API_ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"topic": "job_1776945123456_ab12cd34_output"}'
```

Example success:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "job_name": "job_1776945123456_ab12cd34",
    "topic": "job_1776945123456_ab12cd34_output",
    "output_path": "/abs/workdir/jobs/job_1776945123456_ab12cd34/output/final.mp4"
  }
}
```

## End-to-End Flow

1. Call preprocess API (`process-by-path` or `process-upload`).
2. Service creates workspace:
   - `WORK_DIR/jobs/{job_name}/source`
   - `WORK_DIR/jobs/{job_name}/frames`
   - `WORK_DIR/jobs/{job_name}/audio`
   - `WORK_DIR/jobs/{job_name}/output`
   - `WORK_DIR/jobs/{job_name}/meta`
3. Service extracts frames/audio, probes metadata, encrypts each frame, publishes to `{job_name}_input`.
4. External worker consumes `{job_name}_input`, processes frames, writes encrypted results to `{job_name}_output`.
5. Call `POST /videos/compose` with full output topic.
6. Service collects frames by `frame_index` until expected count or timeout, reconstructs frames, and composes final video with original fps/bitrate + source audio.
7. API returns absolute `output_path`.

## Error Semantics

Business failures map to deterministic HTTP statuses:

- `400`: invalid request/path/topic/upload payload
- `404`: job/manifest/audio not found
- `409`: topic bootstrap conflict, invalid frame contract, compose timeout/missing frames
- `500`: unexpected FFmpeg/Kafka/runtime failures

## Local Tests

```bash
python -m unittest discover -s tests
```
