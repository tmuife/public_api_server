# media_server

`media_server` is a FastAPI worker-facing service that pairs with `media_client`.

It exposes one business API:

- `POST /videos/process-topic`

The API accepts a Kafka input topic created by `media_client` (`{job_name}_input`), then dispatches an async background worker that:

1. consumes encrypted frames from the input topic,
2. decrypts each frame with AES-256-GCM,
3. runs placeholder frame processing,
4. re-encrypts the result,
5. writes result frames to `{job_name}_output`.

## Features

- ASGI entrypoint: `main:app`
- Async dispatch API:
  - `POST /videos/process-topic`
- Public operational routes:
  - `GET /health`
  - `GET /docs`
  - `GET /openapi.json`
- Bearer auth for business routes
- Deterministic Kafka consumer group:
  - `media-server-{job_name}`
- AES-256-GCM frame decrypt/re-encrypt contract compatible with `media_client`
- Best-effort same-instance duplicate dispatch reuse

## Requirements

- Python 3.12+
- Kafka cluster reachable with configured security settings
- The same AES key configuration used by `media_client`

## Quick Start

```bash
cd media_server
uv sync
cp .env.example .env
```

Configure encryption key for `.env` (choose one option):

Option A (recommended): passphrase + salt

```bash
python - <<'PY'
import base64, os
print("AES_256_GCM_PASSPHRASE=your-memorable-passphrase")
print("AES_256_GCM_KDF_SALT_BASE64=" + base64.b64encode(os.urandom(16)).decode())
print("AES_256_GCM_KDF_ITERATIONS=600000")
PY
```

Option B: raw Base64 key

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

Or:

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

- `MEDIA_API_ACCESS_TOKEN`
- one of:
  - `AES_256_GCM_PASSPHRASE` + `AES_256_GCM_KDF_SALT_BASE64`
  - `AES_256_GCM_KEY_BASE64`
- `KAFKA_BOOTSTRAP_SERVERS`

Optional/commonly used settings:

- `KAFKA_SECURITY_PROTOCOL` (`SASL_SSL` default)
- `KAFKA_SASL_MECHANISM`
- `KAFKA_SASL_CONSUMER_USERNAME`
- `KAFKA_SASL_CONSUMER_PASSWORD`
- `KAFKA_SASL_PRODUCER_USERNAME`
- `KAFKA_SASL_PRODUCER_PASSWORD`
- `KAFKA_SASL_USERNAME` / `KAFKA_SASL_PASSWORD` (legacy fallback for both consumer and producer)
- `KAFKA_SSL_CA_FILE`
- `KAFKA_PRODUCE_TIMEOUT_SECONDS`
- `KAFKA_WAIT_FIRST_FRAME_SECONDS`
- `KAFKA_IDLE_TIMEOUT_SECONDS`
- `KAFKA_MAX_PROCESS_SECONDS`
- `KAFKA_TOPIC_PARTITIONS`
- `KAFKA_TOPIC_REPLICATION_FACTOR`
- `AES_256_GCM_KDF_ITERATIONS`
- `MEDIA_API_HOST`
- `MEDIA_API_PORT`

Kafka consumer clients use the `KAFKA_SASL_CONSUMER_*` identity for reading `{job_name}_input`.
Kafka producer and admin clients use the `KAFKA_SASL_PRODUCER_*` identity for creating/verifying and writing `{job_name}_output`.

## API Example

```bash
curl -X POST "http://127.0.0.1:8000/videos/process-topic" \
  -H "Authorization: Bearer ${MEDIA_API_ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"topic": "job_1776945123456_ab12cd34_input"}'
```

To reprocess the same input topic, pass a new `group_id`:

```bash
curl -X POST "http://127.0.0.1:8000/videos/process-topic" \
  -H "Authorization: Bearer ${MEDIA_API_ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"topic": "job_1776945123456_ab12cd34_input", "group_id": "debug-run-001"}'
```

Example accepted response:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "job_name": "job_1776945123456_ab12cd34",
    "input_topic": "job_1776945123456_ab12cd34_input",
    "output_topic": "job_1776945123456_ab12cd34_output",
    "group_id": "media-server-job_1776945123456_ab12cd34",
    "dispatch_mode": "async",
    "status": "accepted"
  }
}
```

Notes:

- HTTP status is `202 Accepted`.
- The API only confirms dispatch. Final frame-processing success or failure is expressed through messages written to the output topic.
- `topic` must match `^[a-zA-Z0-9._-]+$` and end with `_input`.
- `group_id` is optional and uses the same character rule as `topic`; if omitted it defaults to `media-server-{job_name}`.

Query in-memory task status:

```bash
curl "http://127.0.0.1:8000/videos/process-topic/status?topic=job_1776945123456_ab12cd34_input&group_id=debug-run-001" \
  -H "Authorization: Bearer ${MEDIA_API_ACCESS_TOKEN}"
```

Example status response:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "topic": "job_1776945123456_ab12cd34_input",
    "job_name": "job_1776945123456_ab12cd34",
    "output_topic": "job_1776945123456_ab12cd34_output",
    "group_id": "debug-run-001",
    "dispatch_mode": "async",
    "status": "completed",
    "consumed_count": 120,
    "published_count": 120,
    "success_count": 120,
    "error_count": 0,
    "error_code": null,
    "error_message": null
  }
}
```

## Kafka Message Contract

Input frame messages consumed from `{job_name}_input` must include:

- `job_name`
- `frame_index`
- `nonce_b64`
- `ciphertext_b64`
- `tag_b64`
- `content_type`

Output result messages written to `{job_name}_output` must include:

- `job_name`
- `frame_index`
- `status`
- `nonce_b64`
- `ciphertext_b64`
- `tag_b64`

When `status=error`, `error_code` is also required.

Sensitive field names such as `token`, `password`, `secret`, `master_key`, `wrapped_key`, `key`, and `credential` are rejected from Kafka payload/header validation.

## Runtime Behavior

- Same-topic repeated requests with the same `group_id` on one instance reuse the active in-process task registration when possible.
- Same-topic repeated requests with a new `group_id` start a separate Kafka consumer group and can replay retained input messages.
- Multi-instance single-consumer handling relies on Kafka consumer-group coordination through deterministic `group.id`.
- Session completion is controlled by:
  - first-frame wait timeout
  - idle timeout after at least one valid frame
  - max processing duration

## Local Tests

```bash
python -m unittest discover -s tests
```
