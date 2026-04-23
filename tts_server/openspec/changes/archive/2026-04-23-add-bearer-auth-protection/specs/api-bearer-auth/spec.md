## ADDED Requirements

### Requirement: Service SHALL enforce bearer authentication for non-exempt endpoints
The service MUST validate `Authorization: Bearer <token>` for all non-exempt HTTP endpoints and MUST compare the provided token against configured `API_KEY` using constant-time comparison.

#### Scenario: Missing bearer token on protected endpoint
- **WHEN** a client calls a protected endpoint without `Authorization` header
- **THEN** the service returns `HTTP 401` with `WWW-Authenticate: Bearer`

#### Scenario: Invalid bearer token on protected endpoint
- **WHEN** a client calls a protected endpoint with malformed or incorrect bearer token
- **THEN** the service returns `HTTP 401` with a sanitized auth error message

#### Scenario: Valid bearer token on protected endpoint
- **WHEN** a client calls a protected endpoint with valid bearer token
- **THEN** the request proceeds to endpoint business logic without auth rejection

### Requirement: Service SHALL exempt only health check endpoint from bearer authentication
The service MUST keep `GET /health` publicly reachable without bearer token and MUST keep `/` protected by default.

#### Scenario: Health endpoint without token
- **WHEN** a client calls `GET /health` without `Authorization` header
- **THEN** the endpoint responds successfully without auth rejection

#### Scenario: Root endpoint without token
- **WHEN** a client calls `GET /` without `Authorization` header
- **THEN** the service returns `HTTP 401`

### Requirement: Service SHALL protect documentation and schema endpoints
The service MUST require valid bearer token for `/docs`, `/redoc`, and `/openapi.json`.

#### Scenario: Docs access without token
- **WHEN** a client requests `/docs` or `/openapi.json` without bearer token
- **THEN** the service returns `HTTP 401`

#### Scenario: Docs access with valid token
- **WHEN** a client requests `/docs` or `/openapi.json` with valid bearer token
- **THEN** the service returns the requested documentation resource

### Requirement: Service SHALL fail fast on invalid auth configuration when auth is required
If auth enforcement is configured as required, the service MUST fail startup when `API_KEY` is empty or placeholder-like.

#### Scenario: Required auth with invalid API key
- **WHEN** service starts with auth-required mode and `API_KEY` is empty or placeholder
- **THEN** startup fails with actionable configuration error

#### Scenario: Required auth with valid API key
- **WHEN** service starts with auth-required mode and a non-placeholder `API_KEY`
- **THEN** startup succeeds and auth middleware is active
