## 1. Acceptance Test Baseline

- [x] 1.1 Add API acceptance test module covering `POST /v1/audio/speech` stream success (`stream=true`) with response chunk assertions
- [x] 1.2 Add API acceptance test module covering `POST /v1/audio/speech` non-stream success (`stream=false`) with full payload assertions
- [x] 1.3 Add negative acceptance tests for invalid `model` and empty `input` with 4xx status and readable error message assertions
- [x] 1.4 Add reusable test fixtures/helpers for service URL, auth headers, and request payload generation to keep acceptance cases stable

## 2. Runtime Independence Validation

- [x] 2.1 Add an independence validation workflow that runs minimal startup and API-call checks without `MOSS-TTS-Nano-main` and `Kokoro-FastAPI-master`
- [x] 2.2 Add ONNX-only environment validation step that verifies service callability without `torch` and `torchaudio`
- [x] 2.3 Ensure acceptance output reports scenario-level pass/fail details so failures can be quickly located

## 3. Manual Smoke Tooling

- [x] 3.1 Add a curl smoke script for OpenAI-compatible speech API reachability and success/failure exit signaling
- [x] 3.2 Add an OpenAI SDK smoke script that targets local `base_url` and validates client compatibility for speech generation
- [x] 3.3 Document script prerequisites, required environment variables, and example invocation parameters in script headers or companion notes

## 4. Documentation and Release Gate

- [x] 4.1 Update README with startup commands, API request examples, response-format support (`wav`/`pcm`), and known limitations
- [x] 4.2 Add explicit dependency matrix and runtime-boundary notes clarifying reference directories are not runtime dependencies
- [x] 4.3 Add acceptance execution guide covering automated tests, manual smoke scripts, and expected outputs
- [x] 4.4 Add release precheck instructions requiring acceptance success in the reference-directory-removed condition before delivery
