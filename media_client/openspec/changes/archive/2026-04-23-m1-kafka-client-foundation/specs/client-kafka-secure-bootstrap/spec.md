## ADDED Requirements

### Requirement: Service runtime MUST load required Kafka security configuration
The runtime configuration loader MUST parse Kafka bootstrap/security settings in service mode and fail fast on missing required fields.

#### Scenario: Missing required Kafka env in service startup
- **WHEN** FastAPI service starts with missing required Kafka settings
- **THEN** startup validation MUST fail with field-specific configuration error

#### Scenario: Valid Kafka env in service startup
- **WHEN** all required Kafka settings are present and valid
- **THEN** runtime MUST build a validated Kafka configuration object

### Requirement: Service runtime MUST validate SASL/TLS compatibility preflight
Before creating Kafka clients, runtime MUST validate protocol/mechanism compatibility and certificate path readability.

#### Scenario: Invalid protocol-mechanism combination
- **WHEN** configuration contains unsupported security protocol/mechanism combination
- **THEN** runtime MUST reject startup before network calls

#### Scenario: Invalid TLS CA file in service mode
- **WHEN** `SASL_SSL` is configured but CA file is missing/unreadable
- **THEN** runtime MUST fail with certificate validation error details

### Requirement: Service diagnostics MUST redact sensitive settings
Service diagnostics/log output MUST redact tokens, passwords, and key-like fields.

#### Scenario: Emit diagnostics snapshot
- **WHEN** service outputs config snapshot for diagnostics
- **THEN** sensitive fields MUST be masked and never printed in plaintext
