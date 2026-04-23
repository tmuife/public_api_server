## Why

The service currently exposes all endpoints without enforcing bearer token authentication, even though `API_KEY` exists in environment configuration. This creates an avoidable security gap for deployed environments and makes the API behavior inconsistent with the documented OpenAI-style `Authorization: Bearer ...` usage.

## What Changes

- Enforce bearer token authentication for API routes using `API_KEY` from environment configuration.
- Define explicit authentication policy for public/health endpoints versus protected synthesis and metadata endpoints.
- Standardize auth failure behavior (HTTP 401 + `WWW-Authenticate: Bearer`) for missing, malformed, or invalid tokens.
- Add startup-time safety checks for invalid production auth configuration (for example placeholder or empty key when auth is required).
- Update tests and smoke coverage to validate protected and exempt routes.
- Update README auth documentation so request examples and security behavior are aligned.

## Capabilities

### New Capabilities
- `api-bearer-auth`: Centralized bearer token authentication and route-level protection policy for the FastAPI service.

### Modified Capabilities
- `openai-compatible-speech-api`: Require valid bearer token for `/v1` OpenAI-compatible endpoints and define 401 behavior for unauthorized requests.

## Impact

- Affected code:
  - `main.py` route registration and app startup wiring
  - `app/routers/openai_compatible.py`
  - new auth module under `app/` (for token parsing/validation and policy)
  - tests and smoke scripts that currently assume unauthenticated access
  - `README.md` API/auth documentation
- Affected APIs:
  - `/v1/models`, `/v1/audio/voices`, `/v1/audio/speech` expected to require bearer token
  - `/tts/batch`, `/tts/stream` expected to require bearer token
  - `/health` is exempt by policy; `/` and docs/schema endpoints remain protected
- Operational impact:
  - deployments must provide a non-placeholder `API_KEY`
  - unauthorized calls now fail fast with 401
