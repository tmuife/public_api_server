## ADDED Requirements

### Requirement: Service MUST provide an async topic-processing API
The FastAPI service MUST expose `POST /videos/process-topic`, accept a full Kafka input topic name, and return an accepted response without waiting for the entire background processing session to finish.

#### Scenario: Accepted dispatch request
- **WHEN** caller submits a valid input topic such as `job_1776945123456_ab12cd34_input`
- **THEN** service MUST accept the request, schedule background processing, and return HTTP `202 Accepted`

#### Scenario: Reject empty topic
- **WHEN** caller submits an empty topic field
- **THEN** service MUST return HTTP 400 and MUST NOT start background processing

### Requirement: Topic-processing API MUST only accept input topics
The API MUST reject topic names that violate the allowed character set or do not use the `_input` suffix.

#### Scenario: Reject invalid topic characters
- **WHEN** caller submits a topic containing characters outside `^[a-zA-Z0-9._-]+$`
- **THEN** service MUST reject the request with HTTP 400

#### Scenario: Reject non-input topic
- **WHEN** caller submits a topic that does not end with `_input`
- **THEN** service MUST reject the request with HTTP 400

### Requirement: Accepted response MUST describe the derived processing context
Successful dispatch responses MUST return the derived `job_name`, `input_topic`, `output_topic`, deterministic `group_id`, async dispatch mode, and accepted status in the standard success envelope.

#### Scenario: Accepted response payload
- **WHEN** a valid topic-processing request is accepted
- **THEN** response MUST include `code=0`, `message="success"`, and `data.job_name`, `data.input_topic`, `data.output_topic`, `data.group_id`, `data.dispatch_mode`, and `data.status`
