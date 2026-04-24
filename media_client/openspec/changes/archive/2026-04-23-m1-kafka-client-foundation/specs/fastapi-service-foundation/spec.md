## ADDED Requirements

### Requirement: media_client MUST expose a FastAPI application entrypoint
`media_client/main.py` MUST expose a FastAPI `app` object that can be started by ASGI servers.

#### Scenario: Start service via uvicorn
- **WHEN** operator runs `uvicorn main:app --host <host> --port <port>`
- **THEN** the FastAPI application MUST start successfully and accept HTTP requests

### Requirement: Service MUST provide baseline operational routes
The FastAPI service MUST provide operational routes for health and OpenAPI documentation access.

#### Scenario: Health route availability
- **WHEN** client requests `/health`
- **THEN** service MUST return a healthy response with HTTP 200

#### Scenario: OpenAPI route availability
- **WHEN** client requests `/openapi.json`
- **THEN** service MUST return a valid OpenAPI JSON document
