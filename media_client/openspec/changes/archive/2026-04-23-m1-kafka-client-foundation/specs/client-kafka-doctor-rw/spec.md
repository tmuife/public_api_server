## ADDED Requirements

### Requirement: Doctor capability MUST be callable from FastAPI service context
Doctor logic MUST be implemented as service-layer capability callable by protected API endpoint, not only CLI command path.

#### Scenario: Trigger doctor through API
- **WHEN** authorized caller invokes doctor API endpoint
- **THEN** service MUST execute doctor logic and return structured diagnostics response

### Requirement: Doctor API MUST preserve non-zero failure semantics
Doctor failures MUST propagate as unsuccessful API outcomes with explicit stage failure details.

#### Scenario: Kafka metadata failure in API doctor
- **WHEN** doctor cannot access required topic metadata
- **THEN** response MUST include failed stage and `acl_error`-class diagnostics

#### Scenario: Kafka read/write probe failure in API doctor
- **WHEN** produce/consume probe fails in timeout or auth phase
- **THEN** response MUST include failed stage, deterministic error code, and remediation hint
