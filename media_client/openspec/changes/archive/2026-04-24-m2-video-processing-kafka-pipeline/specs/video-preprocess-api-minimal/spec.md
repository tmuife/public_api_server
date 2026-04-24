## ADDED Requirements

### Requirement: Service MUST provide a path-based video preprocess API
The FastAPI service MUST expose `POST /videos/process-by-path` that accepts a server-local video path and starts one preprocess job.

#### Scenario: Process video by local path
- **WHEN** caller submits a valid readable video path
- **THEN** service MUST run the preprocess pipeline and return HTTP 200 with `code=0` and job/topic metadata

#### Scenario: Reject unreadable path
- **WHEN** caller submits a non-existent or unreadable video path
- **THEN** service MUST return HTTP 400 and MUST NOT create a preprocess job

### Requirement: Service MUST provide an upload-based video preprocess API
The FastAPI service MUST expose `POST /videos/process-upload` that accepts a video file upload, stores it to configured upload location, and starts one preprocess job.

#### Scenario: Process uploaded video
- **WHEN** caller uploads a valid video file
- **THEN** service MUST persist the upload to configured directory and run the same preprocess pipeline as path-based API

#### Scenario: Reject invalid upload payload
- **WHEN** caller sends no file or unsupported content
- **THEN** service MUST return HTTP 400 with validation error details

### Requirement: Preprocess APIs MUST share one normalized processing pipeline
Both preprocess endpoints MUST execute identical downstream steps after input normalization: job initialization, frame extraction, frame encryption, and Kafka publication.

#### Scenario: Consistent output contract
- **WHEN** two jobs are submitted through different preprocess endpoints
- **THEN** both responses MUST include the same output schema fields: `job_name`, `input_topic`, and manifest summary

### Requirement: Successful API responses MUST follow unified schema
For preprocess success responses, payload format MUST be `{ code: 0, message: "success", data: ... }`.

#### Scenario: Successful preprocess response shape
- **WHEN** preprocess pipeline completes request handling successfully
- **THEN** response body MUST include top-level `code`, `message`, and `data` fields with `code=0`
