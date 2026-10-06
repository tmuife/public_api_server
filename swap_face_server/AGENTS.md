# Repository Guidelines

## Project Structure & Module Organization
This repository is a FastAPI-based face-swap service.

- `main.py`: local entrypoint (`uvicorn.run(...)`).
- `app/api.py`: FastAPI app wiring, middleware, router registration, websocket endpoint.
- `app/routers/`: HTTP route modules (`*_router.py`, `secure.py`, `template.py`).
- `app/services/`: business/service layer for model inference and media processing.
- `app/middleware/`: auth middleware.
- `models/`: model assets/checkpoints.
- `uploads/`: runtime input/output examples.
- `docker/`, `docker-compose.yml`: container build and local deployment.
- `app/tmp/` and `test.py`: experimental scripts; keep production logic in `app/services` and `app/routers`.

## Build, Test, and Development Commands
Use `uv` for dependency/runtime management.

- `uv sync --frozen --no-dev`: install pinned dependencies from `uv.lock`.
- `uv run python main.py`: run API locally (reads `.env` values such as `APP_HOST`, `APP_PORT`).
- `uv run pytest -q`: run tests (if adding new tests, prefer deterministic, fixture-driven tests).
- `docker compose up --build`: build and run service container on port `8000`.

## Coding Style & Naming Conventions
- Python 3.10, 4-space indentation, UTF-8.
- Follow PEP 8 and type annotations for new/changed code.
- Use `snake_case` for files/functions/variables; `PascalCase` for classes.
- Keep routers thin; move heavy logic into `app/services/`.
- Prefer structured logging over `print()` in production paths.

## Testing Guidelines
- Test framework: `pytest` (with `pytest-asyncio` available for async tests).
- New tests should be named `test_<feature>.py` and avoid machine-specific paths.
- Place repeatable tests under a dedicated `tests/` directory; treat `app/tmp/` as non-CI scratch space.
- Cover API contracts, service edge cases, and failure paths (invalid input, missing files, model load errors).

## Commit & Pull Request Guidelines
Current history uses short, imperative messages (for example: `add docs`, `update`, `... changed to uv`).

- Preferred commit format: `<scope>: <action>` (example: `swap-face: validate websocket payload`).
- Keep commits focused; avoid mixing refactor + feature + infra in one commit.
- PRs should include: purpose, changed modules, local test command/results, config changes (`.env`, Docker), and sample request/response when API behavior changes.

## Security & Configuration Tips
- Do not commit secrets; use `.env` (`API_KEY`, host/port, SSL paths).
- Keep certificates in `certs/` for local/dev only unless explicitly approved.
- Validate uploaded file types/sizes and handle model/runtime exceptions explicitly.
