## 1. FastAPI Foundation and Configuration

- [x] 1.1 Create `media_server/app/` module layout (`routers/services/utils/middleware`) and expose `main.py` as the FastAPI entrypoint
- [x] 1.2 Implement decouple-based settings loader with env-first behavior for API auth, AES key material, Kafka security, and processing window controls
- [x] 1.3 Add startup validation for required Kafka/AES settings and register public `/health`, `/docs`, `/openapi.json` routes
- [x] 1.4 Add Bearer auth protection for `POST /videos/process-topic` and ensure OpenAPI declares the HTTP Bearer scheme

## 2. Async Topic Dispatch API

- [x] 2.1 Implement `POST /videos/process-topic` request/response models with unified success envelope and HTTP `202 Accepted`
- [x] 2.2 Implement input topic validation (`^[a-zA-Z0-9._-]+$` and `_input` suffix) plus derived `job_name`, `output_topic`, and deterministic `group_id`
- [x] 2.3 Implement in-process background task registration so repeated dispatch of the same topic on one instance reuses the existing task instead of starting a duplicate worker
- [x] 2.4 Return accepted response payload containing `job_name`, `input_topic`, `output_topic`, `group_id`, `dispatch_mode`, and accepted status

## 3. Kafka Frame Worker Pipeline

- [x] 3.1 Implement Kafka topic ensure/create for `{job_name}_output` before first result publication
- [x] 3.2 Implement consumer creation using deterministic Kafka `group.id`, manual offset commit, and per-message JSON decoding
- [x] 3.3 Validate incoming frame request payloads against the shared encrypted-frame contract and ignore payloads whose `job_name` does not match the derived topic job
- [x] 3.4 Implement AES-256-GCM frame decryption, placeholder frame processor abstraction, and AES-256-GCM re-encryption for output frames
- [x] 3.5 Publish `status=ok` and `status=error` result messages to `{job_name}_output`, committing offsets only after the corresponding result message is successfully published

## 4. Processing Window Control and Failure Semantics

- [x] 4.1 Implement first-frame wait timeout, idle timeout, and max processing duration controls for background sessions
- [x] 4.2 Emit encrypted error result messages with machine-readable `error_code` when frame decrypt/process/encrypt steps fail
- [x] 4.3 Record structured runtime summaries for accepted, running, completed, and failed background tasks without exposing secrets
- [x] 4.4 Distinguish request-time HTTP failures from background execution failures in logs and service responses

## 5. Validation, Documentation, and OpenSpec Hygiene

- [x] 5.1 Add unit tests for config loading, topic/job/group derivation, AES-256-GCM contract validation, and request auth behavior
- [x] 5.2 Add API/service tests for async dispatch success, duplicate dispatch reuse, invalid topic rejection, and worker-side error publication behavior
- [x] 5.3 Update `media_server/README.md` with environment variables, async dispatch semantics, and example `/videos/process-topic` calls
- [x] 5.4 Run `openspec validate` for `media_server` change artifacts if local OpenSpec project structure is present; otherwise document the validation gap
