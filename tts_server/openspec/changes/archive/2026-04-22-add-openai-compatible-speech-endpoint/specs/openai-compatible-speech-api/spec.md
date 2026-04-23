## ADDED Requirements

### Requirement: Service SHALL provide OpenAI-compatible speech creation endpoint
The API MUST expose `POST /v1/audio/speech` in the current project and MUST accept OpenAI-compatible core request fields: `model`, `input`, `voice`, `response_format`, and `stream`.

#### Scenario: Valid minimal speech request
- **WHEN** the client sends `POST /v1/audio/speech` with valid `model`, non-empty `input`, and valid `voice`
- **THEN** the endpoint accepts the request and returns synthesized audio content

#### Scenario: Missing required field
- **WHEN** the client omits a required field from `POST /v1/audio/speech`
- **THEN** the endpoint returns a 4xx parameter error with a human-readable message

### Requirement: Voice resolution SHALL support builtin and reference-audio sources
The endpoint MUST support two voice sources in the first version: builtin voice names and reference audio provided by upload or reference (`reference_audio_file`, `reference_audio_url`, `reference_audio_path`). When both a builtin `voice` and any reference-audio source are provided, the system MUST prioritize reference audio.

#### Scenario: Builtin voice synthesis
- **WHEN** the client provides a valid builtin `voice` and does not provide reference-audio fields
- **THEN** synthesis uses the builtin voice path

#### Scenario: Reference audio overrides builtin voice
- **WHEN** the client provides both a valid builtin `voice` and a valid reference-audio source
- **THEN** synthesis uses the reference-audio path as higher priority

### Requirement: Endpoint SHALL support both stream and non-stream response modes
For the same valid request contract, `stream=true` MUST return a continuously readable audio chunk stream and `stream=false` MUST return a complete audio payload in a single response.

#### Scenario: Stream mode response
- **WHEN** the client sends `POST /v1/audio/speech` with `stream=true`
- **THEN** the response is streamed as audio chunks until synthesis completes

#### Scenario: Non-stream mode response
- **WHEN** the client sends `POST /v1/audio/speech` with `stream=false`
- **THEN** the response contains the full synthesized audio payload in one response body

### Requirement: Response format SHALL be restricted to wav and pcm in V1
The endpoint MUST accept only `response_format=wav` or `response_format=pcm` in the first version. Any other format value MUST return `HTTP 400`.

#### Scenario: Supported response formats
- **WHEN** the client sets `response_format` to `wav` or `pcm`
- **THEN** the endpoint returns synthesized audio with matching media semantics

#### Scenario: Unsupported response format
- **WHEN** the client sets `response_format` to an unsupported value such as `mp3`
- **THEN** the endpoint returns `HTTP 400` with a readable validation error

### Requirement: Service SHALL provide minimal OpenAI-compatible model discovery
The API MUST expose `GET /v1/models` and MUST return a model list payload that allows OpenAI SDK clients to discover at least one speech-capable model served by the current project.

#### Scenario: Model list retrieval
- **WHEN** the client sends `GET /v1/models`
- **THEN** the endpoint returns a successful model list including at least one valid speech model identifier

### Requirement: Error semantics SHALL distinguish request errors from inference failures
The OpenAI-compatible endpoints MUST return 4xx for invalid client input (including invalid `model`, empty `input`, and unsupported `response_format`) and MUST return 5xx for internal inference/runtime failures.

#### Scenario: Client parameter error classification
- **WHEN** the client sends an invalid `model` or empty `input`
- **THEN** the endpoint returns a 4xx error and an actionable message for client correction

#### Scenario: Internal inference failure classification
- **WHEN** synthesis fails due to backend runtime or inference errors
- **THEN** the endpoint returns a 5xx error with sanitized failure information

### Requirement: Runtime path SHALL remain independent from reference repositories
The runnable implementation for `/v1/audio/speech` and `/v1/models` MUST remain within the current project and MUST NOT require runtime imports from `MOSS-TTS-Nano-main` or `Kokoro-FastAPI-master`.

#### Scenario: Service behavior after reference directory removal
- **WHEN** `MOSS-TTS-Nano-main` and `Kokoro-FastAPI-master` are removed from the repository
- **THEN** the service still starts and both OpenAI-compatible endpoints remain callable
