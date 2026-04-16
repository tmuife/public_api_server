# AGENT.md

## Project Overview

This repository is a multi-service FastAPI workspace. Each subdirectory is an independent API server with a similar scaffold:

- `Florence2_server`
- `embedding_server`
- `enhance_face_server`
- `minicpm_server`
- `owl_vit_server`
- `splade_embedding_server`
- `swap_face_server`

Each service typically contains:

- `main.py`: Uvicorn startup entry
- `app/api.py`: FastAPI app creation and router registration
- `app/routers/`: HTTP routes
- `app/services/`: business/service logic
- `app/middleware/`: middleware (including API key auth)
- `docker/`: container files (`Dockerfile`, `docker-compose.yml`)
- `.env.example`: runtime configuration template

## How To Run

### Option 1: Run one service directly

Example with Florence2 server:

```bash
cd Florence2_server
python main.py
```

### Option 2: Start services with helper script

```bash
bash start.sh
```

`start.sh` currently starts virtualenv-based processes for:

- `Florence2_server`
- `embedding_server`

## Core Parameters

Common environment variables (from `.env.example`):

- `APP_HOST`: service bind host (for example `0.0.0.0`)
- `APP_PORT`: service bind port (for example `8000`)
- `API_KEY`: API key used by auth middleware
- `SSL_CERTIFICATE`: SSL certificate path (optional HTTPS run)
- `SSL_KEYFILE`: SSL private key path (optional HTTPS run)

## API Docs

For a running service (example on port 8000):

- Swagger UI: `http://localhost:8000/docs`
- Sub API docs: `http://localhost:8000/subapi/docs`
- OpenAPI JSON: `http://localhost:8000/api/v1/openapi.json`

## Development Notes

- Keep router files focused on HTTP contracts.
- Keep business logic in `app/services`.
- Add new capability by extending `routers` + `services` consistently.
- If editing one server, verify whether the same change should be mirrored to sibling servers with shared scaffolding.
