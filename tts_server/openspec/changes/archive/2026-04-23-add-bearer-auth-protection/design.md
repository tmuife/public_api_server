## Context

Current FastAPI routes are accessible without authentication, including synthesis endpoints and OpenAI-compatible `/v1` routes, while environment files already define `API_KEY`. This creates a mismatch between expected bearer-token usage and actual runtime behavior.

The change is cross-cutting because it affects:
- app initialization and route protection behavior
- OpenAI-compatible router responses and integration assumptions
- native `/tts/*` endpoints
- docs/openapi exposure policy
- automated tests and smoke scripts

Constraints:
- Keep existing API payload schemas unchanged.
- Preserve health monitoring usability.
- Keep local development workflow practical while preventing insecure production defaults.

## Goals / Non-Goals

**Goals:**
- Enforce bearer token authentication for protected API endpoints using `API_KEY`.
- Provide centralized, consistent auth behavior across `/v1/*` and `/tts/*` routes.
- Return predictable unauthorized responses (`401` with `WWW-Authenticate: Bearer`).
- Define explicit exempt-path policy instead of implicit public exposure.
- Add configuration safeguards to avoid accidental insecure deployment.

**Non-Goals:**
- Introducing OAuth/JWT or external identity providers.
- Multi-tenant key management or scoped permissions.
- Rate limiting, quota management, or advanced abuse controls.
- Reworking endpoint request/response formats beyond auth semantics.

## Decisions

1. Use centralized HTTP middleware for bearer auth enforcement.
- Decision: Add a single middleware to validate authorization header and token for all requests, with explicit exempt paths.
- Rationale: Prevent route-by-route drift and ensure future endpoints are protected by default.
- Alternatives considered:
  - Per-route `Depends`/`Security` dependencies: rejected due to higher risk of accidental omission.
  - Router-level dependencies only: better than per-route but still excludes non-router endpoints (`/docs`, `/openapi.json`) unless separately handled.

2. Define explicit auth policy with minimal default exemptions.
- Decision: Exempt only `/health`; all other endpoints (including `/`) require bearer token.
- Rationale: Keeps observability simple while minimizing accidental public surface.
- Alternatives considered:
  - Exempt all metadata endpoints (`/v1/models`, `/v1/audio/voices`): rejected because these still reveal service capabilities.
  - Exempt docs by default: rejected for production hardening.
  - Exempt root service info (`/`): rejected to keep non-health surface protected by default.

3. Protect API documentation and schema endpoints.
- Decision: Keep `/docs`, `/redoc`, and `/openapi.json` behind bearer authentication in production mode.
- Rationale: API schema and interactive docs expose endpoint surface and payload structure that should not be public by default.
- Alternatives considered:
  - Disable docs entirely in production: not selected in this change to preserve operational troubleshooting convenience.
  - Keep docs public: rejected for security hardening.

4. Standardize token source and comparison logic.
- Decision: Use only `Authorization: Bearer <token>` and compare token with `API_KEY` via constant-time compare.
- Rationale: Aligns with existing client examples and reduces header ambiguity.
- Alternatives considered:
  - Support both bearer and custom header (`X-API-Key`): rejected initially to keep contract clear and avoid mixed client behavior.

5. Enforce configuration safety at startup.
- Decision: Add auth-required configuration check and fail startup when auth is required but key is empty/placeholder.
- Rationale: Avoid deploying a "protected" service with ineffective credentials.
- Alternatives considered:
  - Allow startup and only warn logs: rejected because warnings are easy to miss and unsafe by default.

6. Preserve current API response shapes for authorized requests.
- Decision: Keep success payload formats unchanged; unauthorized requests fail before endpoint logic runs.
- Rationale: Limits integration impact and keeps change focused on access control.

7. Keep streaming behavior unchanged after successful auth.
- Decision: Authenticate request before entering synthesis logic; after auth passes, keep existing streaming/non-streaming behavior as-is.
- Rationale: Avoid introducing latency/behavior regressions unrelated to auth scope.

8. Exclude CIDR allow-list and token-rotation from current scope.
- Decision: Do not add source CIDR allow-list for health probes and do not implement dual-key rotation in this change.
- Rationale: Keep this iteration focused on baseline bearer protection with minimal complexity and fast rollout.
- Alternatives considered:
  - Add CIDR allow-list now: deferred because infrastructure-specific and not required for baseline control.
  - Add dual-key rotation now: deferred to avoid expanding config/state management scope.

## Risks / Trade-offs

- [Misconfigured `API_KEY` breaks service startup] -> Provide clear startup error message and `.env.example` guidance.
- [Unexpected client breakage due to new 401 requirements] -> Update README + smoke scripts + tests with explicit auth headers.
- [Overly strict exemptions block monitoring probes] -> Keep `/health` exempt by default.
- [Default docs exposure may still leak schema if exempted] -> Keep docs protected by default or disable in production profile.
- [Middleware ordering interactions] -> Register auth middleware early and validate with endpoint-level tests including stream paths.

## Migration Plan

1. Introduce auth configuration model and middleware with exempt-path support.
2. Enable middleware in app creation before route handling.
3. Update tests to cover:
   - protected endpoints without token -> 401
   - invalid bearer token -> 401
   - valid token -> existing behavior unchanged
   - exempt endpoint(s) accessible without token
4. Update smoke scripts and README examples to include required auth behavior.
5. Rollout sequence:
   - deploy with `AUTH_REQUIRED=false` (optional transition window)
   - verify clients send bearer token
   - switch to `AUTH_REQUIRED=true`
6. Rollback strategy:
   - temporary rollback via config flag (`AUTH_REQUIRED=false`) without code revert
   - if needed, revert middleware integration commit

## Open Questions

None for this phase.

Resolved decisions:
- `/health` is exempt from bearer authentication.
- `/` is protected by default.
- `/docs`, `/redoc`, and `/openapi.json` are protected.
- CIDR allow-list for health probes is out of scope.
- Dual-key token rotation is out of scope.
