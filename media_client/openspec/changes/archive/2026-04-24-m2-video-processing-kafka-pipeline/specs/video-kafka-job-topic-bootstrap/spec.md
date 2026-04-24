## ADDED Requirements

### Requirement: Service MUST generate deterministic-format unique job names
For each preprocess request, service MUST generate a unique `job_name` using format `job_{utc_millis}_{8char_random}`.

#### Scenario: Generate job name
- **WHEN** service accepts a new preprocess request
- **THEN** generated `job_name` MUST match pattern `^job_[0-9]{13}_[a-z0-9]{8}$`

### Requirement: Input topic MUST be derived from job name
Preprocess pipeline MUST derive input topic as `{job_name}_input` and use it as the only publication topic for encrypted input frames.

#### Scenario: Derive input topic
- **WHEN** `job_name` is generated for preprocess request
- **THEN** service MUST set input topic to exact string concatenation `{job_name}_input`

### Requirement: Service MUST create input topic before frame publication
Before publishing first encrypted frame message, service MUST ensure the derived input topic exists in Kafka.

#### Scenario: Topic creation success
- **WHEN** Kafka cluster allows topic creation
- **THEN** service MUST create or confirm input topic and continue frame publication

#### Scenario: Topic creation failure
- **WHEN** topic creation fails due to ACL or broker error
- **THEN** preprocess request MUST fail and return classified error without publishing partial frame stream

### Requirement: Frame publication MUST preserve index-addressable stream
Service MUST publish one Kafka message per frame with frame index and encrypted payload fields so downstream workers can consume by topic stream.

#### Scenario: Publish encrypted frame stream
- **WHEN** preprocess runs on N extracted frames
- **THEN** service MUST publish N frame messages with deterministic zero-based `frame_index` values
