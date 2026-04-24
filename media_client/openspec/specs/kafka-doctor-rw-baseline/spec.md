# kafka-doctor-rw-baseline Specification

## Purpose
TBD - created by archiving change m1-kafka-client-foundation. Update Purpose after archive.
## Requirements
### Requirement: Doctor API MUST execute staged Kafka diagnostics
The doctor capability MUST execute staged checks and return machine-readable results.

#### Scenario: Run doctor diagnostics
- **WHEN** authorized client triggers doctor execution
- **THEN** response MUST include stage results for config, metadata, and read/write checks

### Requirement: Doctor MUST verify Kafka by real produce/consume probe
Doctor MUST publish and consume a probe message on `KAFKA_TOPIC_DOCTOR_RW` within timeout.

#### Scenario: Real probe success
- **WHEN** probe message is produced and consumed back within timeout
- **THEN** doctor MUST mark read/write stage as pass

#### Scenario: Real probe timeout
- **WHEN** probe consume does not complete before timeout
- **THEN** doctor MUST mark read/write stage as fail with `rw_timeout`

### Requirement: Doctor failures MUST be classified deterministically
Doctor MUST map errors to deterministic categories for operator triage.

#### Scenario: Authentication failure
- **WHEN** broker rejects SASL credentials
- **THEN** doctor MUST return stage failure with `auth_error` and remediation hint

#### Scenario: Authorization failure
- **WHEN** topic access is denied by ACL
- **THEN** doctor MUST return stage failure with `acl_error` and remediation hint

