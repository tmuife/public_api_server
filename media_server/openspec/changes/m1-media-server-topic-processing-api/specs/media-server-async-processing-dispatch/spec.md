## ADDED Requirements

### Requirement: Topic-processing requests MUST dispatch background work asynchronously
The API layer MUST submit topic-processing work to a background execution path and MUST NOT block the caller until the full frame-processing session completes.

#### Scenario: Immediate accepted response
- **WHEN** a valid topic-processing request is received
- **THEN** service MUST return an accepted response without waiting for the worker to finish consuming the topic

### Requirement: Same-instance duplicate dispatches MUST be best-effort deduplicated
If the same input topic is dispatched repeatedly on one service instance while a background task is already active for that topic, the service MUST reuse the active task registration instead of starting a second same-instance worker.

#### Scenario: Duplicate request on same instance
- **WHEN** caller sends `POST /videos/process-topic` twice for the same input topic on the same service instance while the first background task is still active
- **THEN** service MUST return an accepted response and MUST NOT create a second same-instance background worker

### Requirement: Background dispatch MUST retain minimal runtime diagnostics
The service MUST keep a minimal runtime summary for each accepted background task so the instance can log or inspect whether a task is accepted, running, completed, or failed.

#### Scenario: Background task state transition
- **WHEN** a dispatched background task starts processing frames and later exits
- **THEN** the runtime summary MUST reflect at least accepted/running/final state transitions for that task
