## ADDED Requirements

### Requirement: Service SHALL provide authenticated OpenAI-compatible voice listing endpoint
The API MUST expose `GET /v1/audio/voices` and MUST require a valid bearer token for access.

#### Scenario: Authenticated voice list retrieval
- **WHEN** the client sends `GET /v1/audio/voices` with valid bearer token
- **THEN** the endpoint returns a successful voice list payload with canonical voice and alias metadata

#### Scenario: Unauthorized voice list retrieval
- **WHEN** the client sends `GET /v1/audio/voices` without bearer token or with invalid token
- **THEN** the endpoint returns `HTTP 401` and does not expose voice metadata

## MODIFIED Requirements

### Requirement: Service SHALL provide OpenAI-compatible speech creation endpoint
The API MUST expose `POST /v1/audio/speech` in the current project, MUST accept OpenAI-compatible core request fields: `model`, `input`, `voice`, `response_format`, and `stream`, and MUST require valid bearer authentication before synthesis handling.

#### Scenario: Valid authenticated minimal speech request
- **WHEN** the client sends `POST /v1/audio/speech` with valid bearer token, valid `model`, non-empty `input`, and valid `voice`
- **THEN** the endpoint accepts the request and returns synthesized audio content

#### Scenario: Unauthorized speech request
- **WHEN** the client sends `POST /v1/audio/speech` without bearer token or with invalid token
- **THEN** the endpoint returns `HTTP 401` before synthesis execution

#### Scenario: Missing required field after authentication
- **WHEN** the client sends authenticated `POST /v1/audio/speech` but omits a required field
- **THEN** the endpoint returns a 4xx parameter error with a human-readable message

### Requirement: Service SHALL provide minimal OpenAI-compatible model discovery
The API MUST expose `GET /v1/models`, MUST require valid bearer authentication, and MUST return a model list payload that allows OpenAI SDK clients to discover at least one speech-capable model served by the current project.

#### Scenario: Authenticated model list retrieval
- **WHEN** the client sends `GET /v1/models` with valid bearer token
- **THEN** the endpoint returns a successful model list including at least one valid speech model identifier

#### Scenario: Unauthorized model list retrieval
- **WHEN** the client sends `GET /v1/models` without bearer token or with invalid token
- **THEN** the endpoint returns `HTTP 401`

### Requirement: Error semantics SHALL distinguish request errors from inference failures
The OpenAI-compatible endpoints MUST return `HTTP 401` for missing or invalid bearer authentication, MUST return 4xx for invalid client input after successful authentication (including invalid `model`, empty `input`, and unsupported `response_format`), and MUST return 5xx for internal inference/runtime failures.

#### Scenario: Unauthorized request classification
- **WHEN** the client calls an OpenAI-compatible endpoint without valid bearer token
- **THEN** the endpoint returns `HTTP 401` with bearer-auth challenge semantics

#### Scenario: Client parameter error classification
- **WHEN** an authenticated client sends an invalid `model` or empty `input`
- **THEN** the endpoint returns a 4xx error and an actionable message for client correction

#### Scenario: Internal inference failure classification
- **WHEN** an authenticated synthesis request fails due to backend runtime or inference errors
- **THEN** the endpoint returns a 5xx error with sanitized failure information
