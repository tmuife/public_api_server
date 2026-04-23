## Purpose

Define torchless local ONNX runtime requirements for TTS inference in this project.

## Requirements

### Requirement: ONNX runtime SHALL start without PyTorch runtime dependencies
The service runtime MUST start and initialize TTS inference without requiring `torch` or `torchaudio` to be installed in the environment.

#### Scenario: Service startup in torchless environment
- **WHEN** the service is started in an environment where `torch` and `torchaudio` are not installed
- **THEN** the process starts successfully and exposes health endpoints

#### Scenario: ONNX inference initialization without torch import
- **WHEN** inference components are initialized for a synthesis request
- **THEN** initialization succeeds without importing `torch` or `torchaudio`

### Requirement: Runtime implementation SHALL be locally owned by this project
The runnable TTS inference path MUST only depend on modules located in the current project codebase and MUST NOT import runtime modules from `MOSS-TTS-Nano-main` or `Kokoro-FastAPI-master`.

#### Scenario: Import boundary validation
- **WHEN** the service entrypoint and runtime modules are imported
- **THEN** no import path resolves to `MOSS-TTS-Nano-main` or `Kokoro-FastAPI-master`

#### Scenario: Startup after reference directory removal
- **WHEN** the two reference directories are removed from the repository
- **THEN** the service still starts and can initialize the ONNX runtime

### Requirement: Reference audio preprocessing SHALL use torchless resampling
Reference audio loading and resampling for voice cloning MUST be implemented with non-PyTorch libraries, and resampling MUST use `soxr` as the primary resampling backend.

#### Scenario: Non-16k reference audio input
- **WHEN** a reference audio file with a sample rate other than 16kHz is provided
- **THEN** the preprocessing pipeline resamples audio to the model-required rate using `soxr` without `torchaudio`

#### Scenario: Valid cloned prompt conditioning
- **WHEN** a valid reference audio file is provided for synthesis
- **THEN** the system derives conditioning features and proceeds to inference without PyTorch runtime calls

### Requirement: Batch and stream synthesis SHALL return non-empty audio payloads
For the same valid input text and reference audio, both batch and stream synthesis modes MUST produce non-empty audio byte outputs.

#### Scenario: Batch synthesis output
- **WHEN** a batch synthesis request is made with valid text and reference audio
- **THEN** the response contains a non-empty audio byte payload

#### Scenario: Stream synthesis output
- **WHEN** a stream synthesis request is made with valid text and reference audio
- **THEN** the streamed chunks aggregate to a non-empty audio byte payload
