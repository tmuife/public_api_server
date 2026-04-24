## MODIFIED Requirements

### Requirement: Service runtime MUST load required Kafka security configuration
The runtime configuration loader MUST parse Kafka bootstrap/security settings in service mode, include parameters required for dynamic preprocess topic creation and compose consumption, and fail fast on missing required fields.

#### Scenario: Missing required Kafka env in service startup
- **WHEN** FastAPI service starts with missing required Kafka settings
- **THEN** startup validation MUST fail with field-specific configuration error

#### Scenario: Valid Kafka env in service startup
- **WHEN** all required Kafka settings are present and valid
- **THEN** runtime MUST build a validated Kafka configuration object that can be used for topic management and message IO

### Requirement: Service runtime MUST validate SASL/TLS compatibility preflight
Before creating Kafka clients, runtime MUST validate protocol/mechanism compatibility and certificate path readability for configured transport mode.

#### Scenario: Invalid protocol-mechanism combination
- **WHEN** configuration contains unsupported security protocol/mechanism combination
- **THEN** runtime MUST reject startup before network calls

#### Scenario: Invalid TLS CA file in service mode
- **WHEN** `SASL_SSL` is configured but CA file is missing/unreadable
- **THEN** runtime MUST fail with certificate validation error details

### Requirement: Service diagnostics MUST redact sensitive settings
Service diagnostics/log output MUST redact tokens, passwords, and key-like fields, including Kafka credentials and encryption key material.

#### Scenario: Emit diagnostics snapshot
- **WHEN** service outputs config snapshot for diagnostics
- **THEN** sensitive fields MUST be masked and never printed in plaintext

## ADDED Requirements

### Requirement: Service runtime MUST load and validate video workdir and encryption settings
The runtime configuration loader MUST parse `WORK_DIR` and AES-256-GCM key settings and fail fast when they are invalid.

#### Scenario: Missing workdir or invalid AES key
- **WHEN** `WORK_DIR` is absent or AES key is not a valid 32-byte key after decoding
- **THEN** startup validation MUST fail with explicit configuration error

#### Scenario: Valid workdir and AES key
- **WHEN** workdir path is writable and AES key is valid
- **THEN** runtime MUST expose validated settings for preprocess and compose services
