## Why

OpenAI 兼容接口需要在**当前项目**里实现，才能保证后续删除参考目录后仍可持续维护与部署。
因此本 change 的目标是把参考实现思路迁移到当前项目，提供可直接被 OpenAI SDK 调用的本地 `/v1/audio/speech` 能力。

## What Changes

- 在当前项目新增 `POST /v1/audio/speech`，兼容 OpenAI TTS 核心请求字段（至少 `model/input/voice/response_format/stream`）。
- 在 `voice` 侧首版同时支持两类来源：内置 voice 名称，以及上传/引用参考音频（扩展字段：`reference_audio_file`、`reference_audio_url`、`reference_audio_path`）。
- 在同一接口支持 `stream=true`（流式分块）与 `stream=false`（batch 一次返回）。
- 首版 `response_format` 严格限制为 `wav/pcm`，不支持的格式返回 `HTTP 400`。
- 在当前项目新增最小模型发现接口 `GET /v1/models`。
- 统一当前项目错误返回语义，区分参数错误（4xx）与推理错误（5xx）。
- 明确约束：运行时不依赖 `MOSS-TTS-Nano-main` / `Kokoro-FastAPI-master`。

## Capabilities

### New Capabilities
- `openai-compatible-speech-api`: 在当前项目中提供与 OpenAI `audio.speech.create` 兼容的语音生成接口能力。

### Modified Capabilities
- None.

## Impact

- Affected code (current project only):
  - `main.py` 与新增本地路由文件（例如 `app/routers/openai_compatible.py`）
  - ONNX service 层（stream/batch 统一调用）
  - 本地 schema/model 定义模块
- API surface:
  - 新增 `/v1/audio/speech`
  - 新增 `/v1/models`
- Client compatibility:
  - `openai` Python SDK 与 curl（`base_url=http://<host>/v1`）

## Acceptance Criteria

- `POST /v1/audio/speech` 在 `stream=true` 时返回可持续读取的音频分块字节流。
- `POST /v1/audio/speech` 在 `stream=false` 时返回完整音频内容，`Content-Type` 与 `response_format` 匹配。
- `POST /v1/audio/speech` 在仅提供 `voice` 时可走内置 voice 合成；在提供参考音频（上传或引用）时优先使用参考音频完成克隆合成。
- OpenAI SDK 在指向当前项目服务地址时可成功生成音频。
- 非法 `model`、空 `input` 返回明确 4xx 错误与可读信息；不支持的 `response_format` 必须返回 `HTTP 400`。
- 删除 `MOSS-TTS-Nano-main` 与 `Kokoro-FastAPI-master` 后，上述接口行为保持可用。
