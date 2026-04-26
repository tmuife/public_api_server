## ADDED Requirements

### Requirement: Frame request messages MUST carry minimal required fields
Frame request messages consumed by `media_server` MUST include `job_name`, `frame_index`, `nonce_b64`, `ciphertext_b64`, `tag_b64`, and `content_type`.

#### Scenario: Contract-compliant frame request
- **WHEN** worker receives a message with all required request fields
- **THEN** validator MUST accept the message as request-contract compliant

#### Scenario: Missing required request field
- **WHEN** any required request field is omitted
- **THEN** worker MUST reject the message as contract-invalid for processing

### Requirement: Frame result messages MUST include status and encrypted payload
Frame result messages published by `media_server` MUST include `job_name`, `frame_index`, `status`, `nonce_b64`, `ciphertext_b64`, and `tag_b64`.

#### Scenario: Successful frame result payload
- **WHEN** worker completes frame processing successfully
- **THEN** published payload MUST include `status=ok` and all required encrypted payload fields

#### Scenario: Failed frame result payload
- **WHEN** worker classifies a frame as failed
- **THEN** published payload MUST include `status=error`, required encrypted payload fields, and a non-empty `error_code`

### Requirement: Kafka frame messages MUST NOT include secrets
Kafka payloads and headers handled by `media_server` MUST NOT include plaintext keys, wrapped keys, tokens, passwords, or unredacted credential material.

#### Scenario: Forbidden secret field appears
- **WHEN** producer or validator encounters a secret-related field name in payload or headers
- **THEN** publication or processing validation MUST reject that message

### Requirement: Frame index semantics MUST remain deterministic
`frame_index` MUST be zero-based and stable for the lifecycle of one `job_name`.

#### Scenario: Ordered processing
- **WHEN** worker consumes and republishes frames for one job
- **THEN** the frame sequence MUST remain addressable by ascending `frame_index`

### Requirement: Failed frames MUST still be republished as encrypted result messages
If per-frame processing fails after a request message has been accepted for handling, the worker MUST still publish an encrypted result message instead of returning only an out-of-band error.

#### Scenario: Processing failure after valid consume
- **WHEN** worker fails during decrypt, process, or re-encrypt steps for a consumed frame
- **THEN** worker MUST publish a `status=error` result payload that still includes encrypted payload fields and a machine-readable `error_code`
