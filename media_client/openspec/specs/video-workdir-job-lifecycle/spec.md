# video-workdir-job-lifecycle Specification

## Purpose
TBD - created by archiving change m2-video-processing-kafka-pipeline. Update Purpose after archive.
## Requirements
### Requirement: Service MUST manage per-job workspace under configured WORK_DIR
For every job, service MUST create and use `WORK_DIR/jobs/{job_name}` as isolated workspace.

#### Scenario: Create workspace for new job
- **WHEN** a preprocess request is accepted
- **THEN** service MUST create a dedicated job directory under configured `WORK_DIR/jobs`

### Requirement: Job workspace MUST use deterministic subdirectory layout
Each job workspace MUST include deterministic subdirectories for `source`, `frames`, `audio`, `output`, and `meta`.

#### Scenario: Deterministic layout available
- **WHEN** preprocess initializes a new job
- **THEN** all required subdirectories MUST exist before frame extraction begins

### Requirement: Service MUST persist compose-critical manifest metadata
Service MUST persist a manifest file per job containing at least `job_name`, `frame_count`, `fps`, `bitrate`, `audio_path`, and frame format metadata.

#### Scenario: Manifest persisted after preprocess
- **WHEN** frame extraction and metadata probing succeed
- **THEN** service MUST write manifest metadata that can be used by compose without re-probing source video

### Requirement: Service MUST avoid third-party frame storage in job lifecycle
Job lifecycle MUST keep frame intermediate data only in local workspace and Kafka, and MUST NOT require object storage or external frame store.

#### Scenario: No external storage dependency
- **WHEN** preprocess and compose complete successfully
- **THEN** all frame data flow MUST be attributable to local `WORK_DIR` and Kafka topics only

