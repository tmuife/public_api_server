# onnx-api-acceptance-validation Specification

## Purpose
TBD - created by archiving change add-onnx-api-acceptance-tests-and-docs. Update Purpose after archive.
## Requirements
### Requirement: Acceptance suite SHALL validate core ONNX speech API paths
The project MUST provide automated acceptance tests for the OpenAI-compatible speech API that cover at least stream success, non-stream success, invalid model, and empty input scenarios.

#### Scenario: Stream request acceptance passes
- **WHEN** acceptance tests run against a healthy local service and execute a `stream=true` speech request
- **THEN** the suite reports success for stream response behavior and returns readable audio chunks

#### Scenario: Non-stream request acceptance passes
- **WHEN** acceptance tests run against a healthy local service and execute a `stream=false` speech request
- **THEN** the suite reports success for non-stream response behavior and returns a complete audio payload

#### Scenario: Invalid model is rejected as client error
- **WHEN** acceptance tests send a speech request with an unsupported `model` value
- **THEN** the suite verifies the service returns a 4xx status and an actionable error message

#### Scenario: Empty input is rejected as client error
- **WHEN** acceptance tests send a speech request with empty `input`
- **THEN** the suite verifies the service returns a 4xx status and an actionable error message

### Requirement: Acceptance process SHALL verify runtime independence from reference directories
The acceptance process MUST include a runnable check proving that the current project can start and serve minimal OpenAI-compatible speech API calls without runtime dependence on `MOSS-TTS-Nano-main` and `Kokoro-FastAPI-master`.

#### Scenario: Service starts and responds without reference directories
- **WHEN** acceptance executes in an environment where the two reference directories are absent from the repository
- **THEN** service startup succeeds and at least one minimal speech API call completes successfully

#### Scenario: Environment excludes torch and torchaudio for ONNX path
- **WHEN** acceptance executes the ONNX-only runtime path in an environment without `torch` and `torchaudio`
- **THEN** the service remains callable for minimal speech API verification

### Requirement: Project SHALL provide reproducible manual smoke scripts for ONNX speech API
The project MUST include local smoke scripts for both curl-based and OpenAI SDK-based calls so that maintainers can quickly reproduce stream and non-stream checks outside the automated test runner.

#### Scenario: Curl smoke script can validate endpoint reachability
- **WHEN** a maintainer runs the provided curl smoke script with documented prerequisites
- **THEN** the script performs a speech API request and surfaces a clear pass/fail result

#### Scenario: OpenAI SDK smoke script can validate client compatibility
- **WHEN** a maintainer runs the provided OpenAI SDK smoke script with documented base URL and API key settings
- **THEN** the script performs a speech API request through SDK interfaces and surfaces a clear pass/fail result

### Requirement: Documentation SHALL define acceptance contract and runtime boundaries
README and runtime instructions MUST document startup commands, request examples, response format support, dependency matrix, known limitations, and the release precheck that enforces acceptance success after removing reference directories.

#### Scenario: README includes acceptance workflow and support matrix
- **WHEN** a maintainer reviews project documentation after this change
- **THEN** documentation contains executable acceptance steps and explicit support/limitation statements for ONNX speech API usage

#### Scenario: Release precheck requires acceptance success
- **WHEN** a release candidate is prepared for delivery
- **THEN** the documented precheck includes and requires successful acceptance verification in the reference-directory-removed condition

