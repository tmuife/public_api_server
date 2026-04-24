## ADDED Requirements

### Requirement: Frame request messages MUST carry minimal required fields
Frame request messages MUST include `job_id`, `frame_index`, `encrypted_frame_bytes`, `trace_id`, and `attempt`.

#### Scenario: Contract-compliant frame request
- **WHEN** producer submits message with all required fields
- **THEN** validator MUST accept the message as request-contract compliant

#### Scenario: Missing required request field
- **WHEN** any required field is omitted
- **THEN** validator MUST reject the message as contract-invalid

### Requirement: Frame result messages MUST include status and encrypted payload
Frame result messages MUST include `job_id`, `frame_index`, `status`, and `encrypted_frame_bytes`.

#### Scenario: Successful frame result payload
- **WHEN** worker emits success result
- **THEN** payload MUST include `status=ok` and encrypted frame bytes

#### Scenario: Failed frame result payload
- **WHEN** worker emits failure result
- **THEN** payload MUST include `status=error` and machine-readable error classification

### Requirement: Kafka frame messages MUST NOT include secrets
Kafka payload and headers MUST NOT include plaintext keys, wrapped keys, tokens, or passwords.

#### Scenario: Forbidden secret field appears
- **WHEN** producer attempts to include secret-related field
- **THEN** producer-side validation MUST reject publication

### Requirement: Frame index semantics MUST remain deterministic
`frame_index` MUST be zero-based and stable for the lifecycle of one `job_id`.

#### Scenario: Ordered reconstruction
- **WHEN** consumer rebuilds frame sequence for one job
- **THEN** ordering MUST be reconstructable by ascending `frame_index`
