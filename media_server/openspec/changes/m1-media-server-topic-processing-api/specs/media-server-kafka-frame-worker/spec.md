## ADDED Requirements

### Requirement: Background worker MUST consume from the input topic and publish to the derived output topic
For each accepted topic-processing request, the background worker MUST consume encrypted frame messages from `{job_name}_input` and publish processed result messages to `{job_name}_output`.

#### Scenario: Output topic derivation
- **WHEN** service derives `job_name` from an accepted input topic
- **THEN** the worker MUST use `{job_name}_output` as the only result publication topic

#### Scenario: Output topic creation success
- **WHEN** Kafka cluster allows topic creation or topic lookup
- **THEN** worker MUST create or confirm the output topic before first result publication

### Requirement: Kafka worker MUST use deterministic consumer group assignment
The background worker MUST consume using a deterministic Kafka `group.id` derived from the accepted topic context, so multiple service instances participate in one shared consumer group for the same input topic.

#### Scenario: Deterministic group id
- **WHEN** the same input topic is dispatched on two different service instances
- **THEN** both workers MUST compute the same `group.id` value for that topic

#### Scenario: Group-based single-consumer handling
- **WHEN** multiple service instances join the same deterministic `group.id`
- **THEN** Kafka consumer group coordination MUST determine which worker instance handles a given message

### Requirement: Worker MUST commit offsets only after result publication succeeds
The worker MUST use manual offset commits and only commit a consumed frame after the corresponding result message has been successfully published to the output topic.

#### Scenario: Successful publish before commit
- **WHEN** worker consumes a valid frame, publishes its corresponding result message, and receives publish success confirmation
- **THEN** worker MUST commit the consumed offset afterwards

#### Scenario: Publish failure blocks commit
- **WHEN** result publication fails for a consumed frame
- **THEN** worker MUST NOT commit the consumed offset for that frame
