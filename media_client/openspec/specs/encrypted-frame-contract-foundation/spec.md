# encrypted-frame-contract-foundation Specification

## Purpose
TBD - created by archiving change m1-kafka-client-foundation. Update Purpose after archive.
## Requirements
### Requirement: Frame request messages MUST carry minimal required fields
Frame request messages MUST include `job_name`, `frame_index`, `nonce_b64`, `ciphertext_b64`, `tag_b64`, and `content_type`.

#### Scenario: Contract-compliant frame request
- **WHEN** producer submits message with all required fields
- **THEN** validator MUST accept the message as request-contract compliant

#### Scenario: Missing required request field
- **WHEN** any required field is omitted
- **THEN** validator MUST reject the message as contract-invalid

### Requirement: Frame result messages MUST include status and encrypted payload
Frame result messages MUST include `job_name`, `frame_index`, `status`, `nonce_b64`, `ciphertext_b64`, and `tag_b64`.

#### Scenario: Successful frame result payload
- **WHEN** external worker emits success result
- **THEN** payload MUST include `status=ok` and required encrypted payload fields

#### Scenario: Failed frame result payload
- **WHEN** external worker emits failure result
- **THEN** payload MUST include `status=error` and machine-readable error classification

### Requirement: Kafka frame messages MUST NOT include secrets
Kafka payload and headers MUST NOT include plaintext keys, wrapped keys, tokens, passwords, or unredacted credential material.

#### Scenario: Forbidden secret field appears
- **WHEN** producer attempts to include secret-related field
- **THEN** producer-side validation MUST reject publication

### Requirement: Frame index semantics MUST remain deterministic
`frame_index` MUST be zero-based and stable for the lifecycle of one `job_name`.

#### Scenario: Ordered reconstruction
- **WHEN** consumer rebuilds frame sequence for one job
- **THEN** ordering MUST be reconstructable by ascending `frame_index`

### Requirement: Encrypted frame data MUST be transported directly in Kafka messages
Frame transport MUST carry encrypted frame bytes directly in Kafka message payload fields and MUST NOT depend on third-party object storage references.

#### Scenario: Kafka-only frame transport
- **WHEN** preprocess publishes encrypted frames
- **THEN** each frame message MUST include encrypted frame payload fields and MUST NOT require object storage URL dereference

#### Scenario: Reject external storage reference payload
- **WHEN** a frame message provides only external storage location instead of encrypted bytes
- **THEN** message validation MUST fail as contract-invalid

