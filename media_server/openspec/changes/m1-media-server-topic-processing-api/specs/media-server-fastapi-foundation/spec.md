## ADDED Requirements

### Requirement: media_server MUST expose a FastAPI application entrypoint
`media_server/main.py` MUST expose a FastAPI `app` object that can be started by ASGI servers, and the application MUST mount the topic-processing router implemented under `app/` modules.

#### Scenario: Start service via uvicorn
- **WHEN** operator runs `uvicorn main:app --host <host> --port <port>`
- **THEN** the FastAPI application MUST start successfully and accept HTTP requests

#### Scenario: Topic-processing router mounted
- **WHEN** service startup completes
- **THEN** route table MUST include `POST /videos/process-topic`

### Requirement: Service MUST provide baseline operational routes
The FastAPI service MUST provide operational routes for health and OpenAPI documentation access.

#### Scenario: Health route availability
- **WHEN** caller requests `/health`
- **THEN** service MUST return a healthy response with HTTP 200

#### Scenario: OpenAPI route availability
- **WHEN** caller requests `/openapi.json`
- **THEN** service MUST return a valid OpenAPI JSON document

#### Scenario: Docs route availability
- **WHEN** caller requests `/docs`
- **THEN** service MUST return Swagger UI for API inspection and testing

### Requirement: Protected business routes MUST require Bearer authentication
`POST /videos/process-topic` MUST require a valid Bearer token, while operational routes remain public.

#### Scenario: Missing token on protected route
- **WHEN** caller submits `POST /videos/process-topic` without a valid Bearer token
- **THEN** service MUST reject the request with HTTP 401

#### Scenario: Public docs route without token
- **WHEN** caller requests `/docs` without authentication
- **THEN** service MUST still return the documentation UI
