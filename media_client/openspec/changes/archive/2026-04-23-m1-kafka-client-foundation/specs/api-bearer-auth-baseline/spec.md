## ADDED Requirements

### Requirement: Protected APIs MUST enforce Bearer token authentication
All business API endpoints except explicitly exempted routes MUST require `Authorization: Bearer <token>` and validate token against `MEDIA_API_KEY`.

#### Scenario: Missing Authorization header
- **WHEN** client requests a protected endpoint without Authorization header
- **THEN** service MUST return HTTP 401

#### Scenario: Invalid Bearer token
- **WHEN** client requests a protected endpoint with malformed or mismatched token
- **THEN** service MUST return HTTP 401

#### Scenario: Valid Bearer token
- **WHEN** client requests a protected endpoint with valid Bearer token
- **THEN** service MUST authorize the request and continue endpoint execution

### Requirement: Auth exemptions MUST be deterministic
The service MUST keep `/health`, `/docs`, `/openapi.json`, and `/docs/oauth2-redirect` in auth-exempt path set.

#### Scenario: Unauthenticated health check
- **WHEN** client requests `/health` without Authorization header
- **THEN** service MUST return HTTP 200 and MUST NOT require login

#### Scenario: Unauthenticated docs access
- **WHEN** browser requests `/docs` without Authorization header
- **THEN** service MUST return the docs page

### Requirement: OpenAPI MUST advertise Bearer scheme for docs login
OpenAPI schema MUST include HTTP bearer security scheme so `/docs` can authorize requests.

#### Scenario: Docs authorization capability
- **WHEN** user opens `/docs`
- **THEN** Swagger UI MUST provide Bearer authorize control for protected APIs
