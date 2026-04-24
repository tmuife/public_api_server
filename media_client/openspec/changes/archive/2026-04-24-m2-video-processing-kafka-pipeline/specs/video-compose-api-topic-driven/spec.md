## ADDED Requirements

### Requirement: Service MUST provide a topic-driven compose API
The FastAPI service MUST expose `POST /videos/compose` and accept a full Kafka topic name as input.

#### Scenario: Compose request with full topic
- **WHEN** caller submits a full topic string such as `job_1776945123456_ab12cd34_output`
- **THEN** service MUST use the provided topic directly for consumption without internal renaming

#### Scenario: Reject empty topic
- **WHEN** caller submits an empty topic field
- **THEN** service MUST return HTTP 400 and MUST NOT start compose work

### Requirement: Compose flow MUST complete without end-of-stream marker
The compose logic MUST determine completion by expected frame count from job manifest and timeout configuration, not by Kafka end marker message.

#### Scenario: All expected frames received before timeout
- **WHEN** consumer receives unique frame indexes up to expected frame count
- **THEN** service MUST proceed to video compose phase and finish successfully

#### Scenario: Frame collection timeout
- **WHEN** expected frame count is not reached before configured consume timeout
- **THEN** service MUST fail compose request with timeout classification and missing-frame summary

### Requirement: Compose output MUST preserve source timing and bitrate intent
Compose logic MUST reconstruct frames in ascending index order and encode video using original job metadata (`fps`, target bitrate) with source audio track.

#### Scenario: Successful video composition
- **WHEN** all required frames and source audio are available
- **THEN** service MUST output a playable video file with original fps and configured bitrate target

### Requirement: Compose API MUST return output file path
Successful compose response MUST return the absolute path of the composed video file in response `data`.

#### Scenario: Compose success response
- **WHEN** compose operation completes
- **THEN** response MUST include `code=0`, `message="success"`, and `data.output_path`
