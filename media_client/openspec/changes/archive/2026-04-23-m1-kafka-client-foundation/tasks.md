## 1. FastAPI Service Foundation

- [x] 1.1 Refactor `main.py` to expose FastAPI `app` entrypoint for ASGI startup
- [x] 1.2 Introduce FastAPI app factory and lifecycle hooks for startup validation
- [x] 1.3 Add baseline operational routes (`/health`) with unified response schema
- [x] 1.4 Keep or migrate existing CLI behavior into compatibility shim or separate script

## 2. Bearer Authentication Baseline

- [x] 2.1 Implement global Bearer auth middleware/dependency with `MEDIA_API_KEY` validation
- [x] 2.2 Configure explicit auth exemptions for `/health`, `/docs`, `/openapi.json`, `/docs/oauth2-redirect`
- [x] 2.3 Add HTTP Bearer security scheme to OpenAPI generation for docs authorize flow
- [x] 2.4 Standardize unauthorized response handling (`401`) for missing/invalid tokens

## 3. Kafka Secure Bootstrap (Service Mode)

- [x] 3.1 Move Kafka settings loading into service runtime config module
- [x] 3.2 Enforce startup-time validation for required Kafka env fields and value formats
- [x] 3.3 Validate `SASL_SSL`/TLS certificate path readability before Kafka client creation
- [x] 3.4 Add redaction utility to prevent token/password/key leakage in logs and diagnostics

## 4. Doctor Service and API Endpoint

- [x] 4.1 Refactor doctor logic into service-layer callable (not CLI-only path)
- [x] 4.2 Add protected doctor endpoint (recommended: `POST /v1/system/doctor`)
- [x] 4.3 Implement staged Kafka checks (`config`, `kafka.metadata`, `kafka.read_write`)
- [x] 4.4 Implement real produce/consume probe on `KAFKA_TOPIC_DOCTOR_RW`
- [x] 4.5 Implement deterministic error mapping (`config_error`, `network_error`, `auth_error`, `acl_error`, `rw_timeout`)

## 5. Message Contract Guardrails

- [x] 5.1 Implement frame request contract validator (`job_id`, `frame_index`, `encrypted_frame_bytes`, `trace_id`, `attempt`)
- [x] 5.2 Implement frame result contract validator (`job_id`, `frame_index`, `status`, `encrypted_frame_bytes`)
- [x] 5.3 Enforce forbidden secret-field checks for Kafka payload/header publication path
- [x] 5.4 Ensure frame index semantics remain zero-based and deterministic in contract utilities

## 6. Testing and Documentation

- [x] 6.1 Add API tests for auth baseline and route exemptions
- [x] 6.2 Add docs/OpenAPI tests verifying Bearer scheme and authorize path usability
- [x] 6.3 Add config validation tests for Kafka startup checks and redaction behavior
- [x] 6.4 Add doctor API tests for metadata success/failure and real probe failure classification
- [x] 6.5 Update README and `.env.example` to document FastAPI startup, auth model, and doctor API usage
