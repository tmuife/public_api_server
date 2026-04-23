## 1. OpenAI-compatible API contract and models

- [x] 1.1 Add request/response schemas for `POST /v1/audio/speech` covering `model`, `input`, `voice`, `response_format`, `stream`, and optional reference-audio fields (`reference_audio_file`, `reference_audio_url`, `reference_audio_path`).
- [x] 1.2 Add schema-level validation rules for non-empty `input` and allowed `response_format` values (`wav`, `pcm`) with explicit 400-ready error messages.
- [x] 1.3 Define `GET /v1/models` response schema compatible with OpenAI model discovery clients and include at least one speech model identifier.

## 2. Router and service implementation

- [x] 2.1 Create and register a local `/v1` router (for example `app/routers/openai_compatible.py`) in `main.py` without importing runtime logic from reference repositories.
- [x] 2.2 Implement `POST /v1/audio/speech` request handling that routes `stream=true` to streaming output and `stream=false` to full-payload output through a shared local service facade.
- [x] 2.3 Implement voice source resolution in service layer to support builtin voice and reference-audio sources, with deterministic priority: reference-audio source overrides builtin `voice`.
- [x] 2.4 Implement strict `response_format` handling so only `wav/pcm` are accepted and unsupported formats return `HTTP 400`.
- [x] 2.5 Implement `GET /v1/models` endpoint returning the minimal model list payload expected by OpenAI SDK-style clients.
- [x] 2.6 Ensure error mapping distinguishes client input failures (4xx) from synthesis/runtime failures (5xx) with sanitized, human-readable error details.

## 3. Compatibility and acceptance verification

- [x] 3.1 Add API tests for `/v1/audio/speech` stream mode that verify chunked audio output is readable and non-empty.
- [x] 3.2 Add API tests for `/v1/audio/speech` non-stream mode that verify complete audio output and content semantics for `wav` and `pcm`.
- [x] 3.3 Add API tests for voice resolution behavior: builtin-only path works, reference-audio path works, and reference-audio source takes precedence when both are present.
- [x] 3.4 Add negative tests for invalid `model`, empty `input`, and unsupported `response_format` (`HTTP 400`), plus one backend failure case returning `HTTP 5xx`.
- [x] 3.5 Add a compatibility smoke test using OpenAI SDK-style invocation against local `base_url=/v1` for speech generation.

## 4. Documentation and migration checks

- [x] 4.1 Update `README.md` with `/v1/audio/speech` and `/v1/models` usage examples, supported `response_format`, and reference-audio options.
- [x] 4.2 Update `AGENT.md` to document OpenAI-compatible endpoint behavior, error semantics, and voice-source priority.
- [x] 4.3 Run and record validation proving `/v1` endpoints remain functional after temporary removal of `MOSS-TTS-Nano-main` and `Kokoro-FastAPI-master`.
