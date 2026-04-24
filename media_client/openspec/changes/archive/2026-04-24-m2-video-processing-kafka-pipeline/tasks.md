## 1. Configuration and Project Structure

- [x] 1.1 Create `app/` module layout (`routers/services/utils`) and wire imports from `main.py`
- [x] 1.2 Implement decouple-based settings loader with env-first behavior and typed validation
- [x] 1.3 Add required env keys to `.env.example` (`WORK_DIR`, Kafka settings, `AES_256_GCM_KEY_BASE64`, timeout settings)
- [x] 1.4 Add startup validation for writable `WORK_DIR` and valid 32-byte AES-256-GCM key

## 2. Preprocess APIs and Job Bootstrap

- [x] 2.1 Implement `POST /videos/process-by-path` request/response models and validation
- [x] 2.2 Implement `POST /videos/process-upload` upload persistence logic with configurable upload directory
- [x] 2.3 Implement shared preprocess service pipeline for both endpoints
- [x] 2.4 Implement `job_name` generator (`job_{utc_millis}_{8char_random}`) and `{job_name}_input` topic derivation
- [x] 2.5 Implement Kafka topic create-or-ensure step before first frame publish

## 3. Frame Extraction, Encryption, and Kafka Messaging

- [x] 3.1 Implement FFmpeg/FFprobe helpers to extract frames, source audio, and source metadata
- [x] 3.2 Implement AES-256-GCM encrypt/decrypt utilities with nonce/tag handling and base64 encoding
- [x] 3.3 Define and enforce Kafka frame message contract fields for input stream
- [x] 3.4 Publish encrypted frame messages with zero-based `frame_index` and retry-safe error handling
- [x] 3.5 Persist per-job manifest metadata (`frame_count`, `fps`, `bitrate`, audio path, frame format)

## 4. Compose API and Video Reconstruction

- [x] 4.1 Implement `POST /videos/compose` request/response models using full topic input
- [x] 4.2 Implement topic-to-job manifest resolution and compose preflight checks
- [x] 4.3 Implement Kafka consumption loop that collects frames by index until expected frame count or timeout
- [x] 4.4 Implement frame decode/reconstruction and FFmpeg compose using original fps/bitrate plus source audio
- [x] 4.5 Return composed video absolute output path in unified success response

## 5. Error Handling, Validation, and Documentation

- [x] 5.1 Map business failures to deterministic HTTPException responses (400/404/409/500)
- [x] 5.2 Ensure sensitive values are redacted from logs and diagnostics (Kafka creds, AES key material)
- [x] 5.3 Add tests for config precedence, job/topic naming, encrypt/decrypt contract, and API happy/failure paths
- [x] 5.4 Update `README.md` with API usage examples and end-to-end preprocess/compose flow
- [x] 5.5 Run `openspec validate --change m2-video-processing-kafka-pipeline` and resolve all validation issues
