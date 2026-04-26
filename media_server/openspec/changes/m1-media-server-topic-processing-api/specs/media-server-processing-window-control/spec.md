## ADDED Requirements

### Requirement: Worker MUST determine completion without end-of-stream markers
The background worker MUST determine processing-session completion using configured time windows rather than explicit EOF messages or total frame count fields.

#### Scenario: Session ends after idle timeout
- **WHEN** worker has already processed at least one valid frame and no new frame arrives before `KAFKA_IDLE_TIMEOUT_SECONDS`
- **THEN** worker MUST end the session as complete for that run

#### Scenario: Session stops at max duration
- **WHEN** processing runtime reaches `KAFKA_MAX_PROCESS_SECONDS`
- **THEN** worker MUST stop consuming further frames for that run

### Requirement: Worker MUST enforce first-frame wait limits
The background worker MUST fail or abandon a newly dispatched session if no valid frame arrives within the configured first-frame wait window.

#### Scenario: No frame before first-frame timeout
- **WHEN** no valid frame is consumed before `KAFKA_WAIT_FIRST_FRAME_SECONDS`
- **THEN** worker MUST terminate the session and record a timeout-classified failure for that run

### Requirement: Service MUST NOT require protocol changes for completion signaling in this change
This change MUST continue using the existing `media_client` input contract and MUST NOT require EOF/seal messages or total-frame metadata additions to start processing.

#### Scenario: Existing client contract remains sufficient
- **WHEN** caller uses the current `media_client` preprocess output topic
- **THEN** `media_server` MUST be able to start processing without requiring additional completion marker messages
