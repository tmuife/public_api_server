## MODIFIED Requirements

### Requirement: media_client MUST expose a FastAPI application entrypoint
`media_client/main.py` MUST expose a FastAPI `app` object that can be started by ASGI servers, and the application MUST mount the video-processing router implemented under `app/` modules.

#### Scenario: Start service via uvicorn
- **WHEN** operator runs `uvicorn main:app --host <host> --port <port>`
- **THEN** the FastAPI application MUST start successfully and accept HTTP requests

#### Scenario: Video router mounted
- **WHEN** service startup completes
- **THEN** route table MUST include `/videos/process-by-path`, `/videos/process-upload`, and `/videos/compose`

### Requirement: Service MUST provide baseline operational routes
The FastAPI service MUST provide operational routes for health and OpenAPI documentation access, while preserving baseline availability for API debugging.

#### Scenario: Health route availability
- **WHEN** client requests `/health`
- **THEN** service MUST return a healthy response with HTTP 200

#### Scenario: OpenAPI route availability
- **WHEN** client requests `/openapi.json`
- **THEN** service MUST return a valid OpenAPI JSON document

#### Scenario: Docs route availability
- **WHEN** client requests `/docs`
- **THEN** service MUST return Swagger UI page for API inspection and testing
