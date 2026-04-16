# Public API Server Monorepo

A multi-service FastAPI workspace for vision, embedding, OCR, face, and multimodal inference APIs.

This repository contains several independent API servers. Each server has its own `main.py`, app routers, dependencies, and Docker setup.

## What Is In This Repository

- `Florence2_server`: Florence-2 image understanding API
- `embedding_server`: BGE-M3, CLIP, OCR, and face embedding APIs
- `enhance_face_server`: face enhancement/swap oriented server (swap routes enabled)
- `minicpm_server`: MiniCPM multimodal (image/audio question answering)
- `owl_vit_server`: OWL-ViT text-guided object detection
- `splade_embedding_server`: embedding-style server scaffold (currently same router set as `embedding_server`)
- `swap_face_server`: face swap APIs and websocket streaming

## Repository Layout

```text
public_api_server/
├── Florence2_server/
├── embedding_server/
├── enhance_face_server/
├── minicpm_server/
├── owl_vit_server/
├── splade_embedding_server/
├── swap_face_server/
└── start.sh
```

Inside each service:

- `main.py`: Uvicorn startup entry
- `app/api.py`: FastAPI app construction + router registration
- `app/routers/`: endpoint definitions
- `app/services/`: model/service logic
- `docker/`: Dockerfile and docker-compose config

## Service Matrix

| Service | Main Capability | Enabled API Prefixes | Notes |
|---|---|---|---|
| `Florence2_server` | Florence-2 image processing | `/template`, `/secure`, `/florence` | `m3/clip/ocr` routers exist but are disabled in `app/api.py` |
| `embedding_server` | Embeddings + OCR + face features | `/template`, `/secure`, `/bge`, `/clip`, `/ocr`, `/face` | Full embedding stack enabled |
| `enhance_face_server` | Face swap/enhance workflow | `/template`, `/secure`, `/face` | Uses `swap_face_router` |
| `minicpm_server` | Multimodal QA (image/audio) | `/template`, `/secure`, `/minicpm` | MiniCPM Omni-style inference |
| `owl_vit_server` | Text-prompt object detection | `/template`, `/secure`, `/detect` | OWL-ViT model configured by `.env` |
| `splade_embedding_server` | Embedding scaffold server | `/template`, `/secure`, `/bge`, `/clip`, `/ocr`, `/face` | Current routing mirrors `embedding_server` |
| `swap_face_server` | Face swap REST + websocket | `/template`, `/secure`, `/face`, websocket `/ws` | Includes upload and stream paths |

## Security Model

Protected routes use API-key auth via header:

- Header name: `access_token`
- Value: `API_KEY` from `.env`

Example:

```bash
-H "access_token: ${API_KEY}"
```

## Environment Variables

Most services use this baseline:

```env
APP_HOST=0.0.0.0
APP_PORT=8000
API_KEY=your_key
```

Additional service-specific variables:

- `owl_vit_server`: `model_id`, `device`
- `swap_face_server`: `execution_providers`, `detect_method`, `distance`, DB fields, etc.
- `minicpm_server`: benchmark/test helper vars in `.env.example`

## Local Development

## Prerequisites

- Python installed (version depends on service)
- `ffmpeg` available on host (used by multiple services)
- Model files present when required (`models/`, `gfpgan/` for some services)

Python version guidance from service configs:

- Python `^3.12`: `Florence2_server`, `embedding_server`, `splade_embedding_server`
- Python `>=3.10,<3.11`: `enhance_face_server`, `swap_face_server`
- Python `^3.10`: `minicpm_server`
- Python `>=3.10`: `owl_vit_server`

## Run One Service (Poetry-based)

Use this for all services except `owl_vit_server`.

```bash
cd <service_dir>
cp .env.example .env
poetry install
poetry run python main.py
```

Then open:

- `http://127.0.0.1:<APP_PORT>/docs`
- `http://127.0.0.1:<APP_PORT>/subapi/docs`

## Run `owl_vit_server` (uv-based)

```bash
cd owl_vit_server
cp .env.example .env
uv sync
.venv/bin/python main.py
```

## Start Predefined Multi-Service Script

```bash
bash start.sh
```

Current behavior of `start.sh`:

- starts `Florence2_server`
- starts `embedding_server`

## Docker Deployment

Each service has its own compose file:

```bash
cd <service_dir>/docker
docker compose up -d --build
```

Default host port mappings from compose files:

- `Florence2_server`: `5000 -> 8000`
- `minicpm_server`: `5000 -> 8000`
- Most others: `8000 -> 8000`

When running multiple services together, update host ports to avoid conflicts.

## API Examples

## 1) Health-style check

```bash
curl http://127.0.0.1:8000/template/
```

## 2) Secure endpoint check

```bash
curl -H "access_token: ${API_KEY}" \
  http://127.0.0.1:8000/secure/dog
```

## 3) BGE embedding (embedding server)

```bash
curl -X POST "http://127.0.0.1:8000/bge/embeddings/" \
  -H "Content-Type: application/json" \
  -H "access_token: ${API_KEY}" \
  -d '{"sentences": ["hello world", "fastapi"]}'
```

## 4) OWL-ViT detect with prompt

```bash
curl -X POST "http://127.0.0.1:8000/detect/detect_with_prompt" \
  -H "Content-Type: application/json" \
  -H "access_token: ${API_KEY}" \
  -d '{"image_base": "<base64_image>", "texts": "cat|dog"}'
```

## 5) MiniCPM image query

```bash
curl -X POST "http://127.0.0.1:8000/minicpm/image_query" \
  -H "Content-Type: application/json" \
  -H "access_token: ${API_KEY}" \
  -d '{"content": "<base64_image>", "question": "Describe this image"}'
```

## 6) Swap face websocket

`swap_face_server` also exposes websocket stream endpoint:

- `ws://127.0.0.1:8000/ws?access_token=<API_KEY>`

## Notes And Caveats

- Response formats are not fully unified across all routers.
- Many inference endpoints expect base64 payload strings in JSON fields like `content` or `image_base`.
- Several services require large model files that are not auto-downloaded by this README.
- Some compose files share the same host port; adjust before parallel deployment.
- `splade_embedding_server` currently does not provide a `.env.example` at repository root, so create `.env` manually if needed.

## License

See `LICENSE` / `LICENSE.md` in the repository root.
