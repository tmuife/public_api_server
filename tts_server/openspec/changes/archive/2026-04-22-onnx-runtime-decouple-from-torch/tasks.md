## 1. Local runtime module extraction

- [x] 1.1 Create `app/runtime`, `app/services`, and `app/utils` packages with explicit local import boundaries.
- [x] 1.2 Extract ONNX inference runtime code into `app/runtime` so model loading and synthesis no longer import from `MOSS-TTS-Nano-main` or `Kokoro-FastAPI-master`.
- [x] 1.3 Add runtime configuration objects (model paths, sample rate, streaming options) in local modules and wire initialization through the local service layer.

## 2. Torchless audio preprocessing

- [x] 2.1 Update project dependencies to support torchless audio I/O and resampling, and ensure runtime path does not require `torch`/`torchaudio`.
- [x] 2.2 Implement reference audio loading and normalization in `app/utils` using non-PyTorch libraries.
- [x] 2.3 Implement resampling pipeline with `soxr` as the primary backend and integrate it into reference audio preprocessing.
- [x] 2.4 Add a verification case for non-16k reference audio to confirm preprocessing outputs model-ready audio without PyTorch runtime calls.

## 3. Service entrypoint and synthesis path integration

- [x] 3.1 Implement local TTS service orchestration in `app/services` for batch and stream synthesis based on the local ONNX runtime.
- [x] 3.2 Refactor API entrypoint/routes to call only local service modules and keep unified success response schema.
- [x] 3.3 Add a guard check (test or script) that fails if runtime imports resolve to `MOSS-TTS-Nano-main` or `Kokoro-FastAPI-master`.
- [x] 3.4 Ensure service startup and inference initialization succeed when `torch` and `torchaudio` are not installed.

## 4. Acceptance and migration verification

- [x] 4.1 Add acceptance checks proving batch mode returns non-empty audio bytes for valid text plus reference audio.
- [x] 4.2 Add acceptance checks proving stream mode returns non-empty aggregated audio bytes for the same input.
- [x] 4.3 Run smoke validation after removing/moving reference directories to confirm startup and minimal inference still work.
- [x] 4.4 Update `README.md` and `AGENT.md` to document the torchless ONNX runtime path, required dependencies, and validation commands.
