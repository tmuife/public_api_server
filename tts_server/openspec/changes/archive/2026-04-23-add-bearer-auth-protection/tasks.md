## 1. Auth Configuration Foundation

- [x] 1.1 Add auth configuration parsing for `API_KEY` and auth-required mode in app configuration utilities.
- [x] 1.2 Implement startup validation that fails fast when auth is required but `API_KEY` is empty or placeholder-like.
- [x] 1.3 Add unit-level checks for auth configuration parsing and startup validation behavior.

## 2. Bearer Middleware Implementation

- [x] 2.1 Implement centralized HTTP middleware that validates `Authorization: Bearer <token>` using constant-time comparison.
- [x] 2.2 Return standardized unauthorized responses (`HTTP 401` + `WWW-Authenticate: Bearer`) for missing, malformed, or invalid token.
- [x] 2.3 Add explicit exempt-path policy so only `/health` bypasses auth while `/`, `/docs`, `/redoc`, and `/openapi.json` remain protected.

## 3. App Integration And Endpoint Protection

- [x] 3.1 Wire auth middleware into `create_app()` so protection applies consistently to `/v1/*`, `/tts/*`, and docs/schema endpoints.
- [x] 3.2 Verify protected endpoints reject unauthorized requests before reaching synthesis/business logic.
- [x] 3.3 Verify successful authorized requests preserve existing response formats and stream/non-stream behavior.

## 4. Test Coverage Updates

- [x] 4.1 Update OpenAI-compatible API tests to send valid bearer token for success paths and add unauthorized-path assertions (`/v1/models`, `/v1/audio/voices`, `/v1/audio/speech`).
- [x] 4.2 Add tests for route policy: `/health` accessible without token, `/` returns `401` without token, docs/schema endpoints return `401` without token.
- [x] 4.3 Update acceptance helpers and acceptance tests to align with enforced bearer auth and keep existing synthesis assertions intact.

## 5. Scripts And Documentation Alignment

- [x] 5.1 Update smoke scripts and local run instructions to require/configure bearer token explicitly.
- [x] 5.2 Update README auth sections to document protected endpoints, exempt endpoint policy, and unauthorized response semantics.
- [x] 5.3 Update `.env.example` and related notes to clarify secure `API_KEY` expectations for auth-required deployments.

## 6. Validation And Rollout Readiness

- [x] 6.1 Run project test and smoke checks covering both unauthorized and authorized scenarios, including stream path.
- [x] 6.2 Verify migration toggle behavior for rollout (`AUTH_REQUIRED=false` transition, then `AUTH_REQUIRED=true`).
- [x] 6.3 Record final verification evidence in the change workflow before implementation handoff.

## Verification Evidence

- `uv run python -m unittest tests.test_auth_config tests.test_bearer_auth_policy tests.test_openai_compatible_api -v` -> PASS
- `uv run python -m unittest tests.test_onnx_api_acceptance -v` -> PASS
- `./scripts/run_onnx_api_acceptance.sh` -> PASS (3/3 smoke scenarios)
- `uv run python -m unittest discover -s tests -p 'test_*.py' -v` -> PASS (33 tests)
