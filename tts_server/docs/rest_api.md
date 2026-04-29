# TTS Server REST API 接口文档

本文档用于给其它系统对接当前 `tts_server` 的 REST API。

## 1. 基础信息

- 默认基地址：`http://127.0.0.1:8000`
- 在线 OpenAPI（受鉴权影响）：`/openapi.json`
- Swagger UI（受鉴权影响）：`/docs`
- ReDoc（受鉴权影响）：`/redoc`

## 2. 鉴权方式

默认配置下（`AUTH_REQUIRED=true`），除 `GET /health` 外其余接口均需要 Bearer Token。

请求头：

```http
Authorization: Bearer <API_KEY>
```

未携带或 token 非法时：

- HTTP 状态码：`401`
- 响应头：`WWW-Authenticate: Bearer`
- 响应体：`{"detail":"Unauthorized"}`

## 3. 接口总览

| 功能 | 方法 | 路径 | 是否默认鉴权 | 调用方式 |
| --- | --- | --- | --- | --- |
| 服务信息 | GET | `/` | 是 | JSON 响应 |
| 健康检查 | GET | `/health` | 否 | JSON 响应 |
| 批量合成（统一包裹） | POST | `/tts/batch` | 是 | JSON 入参 + JSON 出参 |
| 分片合成（统一包裹） | POST | `/tts/stream` | 是 | JSON 入参 + JSON 出参 |
| 模型列表（OpenAI 兼容） | GET | `/v1/models` | 是 | JSON 响应 |
| 音色列表（OpenAI 兼容） | GET | `/v1/audio/voices` | 是 | JSON 响应 |
| 语音合成（OpenAI 兼容） | POST | `/v1/audio/speech` | 是 | JSON 或 multipart/form-data，返回音频字节流 |
| Swagger 文档页 | GET | `/docs` | 是（可配置放开） | 浏览器访问 |
| ReDoc 文档页 | GET | `/redoc` | 是（可配置放开） | 浏览器访问 |
| OpenAPI Schema | GET | `/openapi.json` | 是（可配置放开） | JSON 响应 |

> 说明：设置 `DOCS_PUBLIC_IN_DEV=true` 时，`/docs`、`/redoc`、`/openapi.json` 可免鉴权（仅建议开发环境）。

## 4. 统一响应结构（适用于 `/`、`/health`、`/tts/*`）

```json
{
  "code": 0,
  "message": "success",
  "data": {}
}
```

- `code=0` 表示成功
- 非 2xx 场景统一走 HTTP 错误码和 `detail`

## 5. 详细接口说明

### 5.1 `GET /`

- 功能：返回服务基础信息
- 调用方式：HTTP GET
- 请求头：`Authorization: Bearer <API_KEY>`（默认需要）

示例响应：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "service": "tts-server",
    "version": "0.2.0",
    "host": "0.0.0.0",
    "port": 8000
  }
}
```

### 5.2 `GET /health`

- 功能：健康检查（可用于探活）
- 调用方式：HTTP GET
- 鉴权：默认免鉴权

示例响应：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": "ok",
    "service": "tts-server"
  }
}
```

### 5.3 `POST /tts/batch`

- 功能：文本转语音，返回一次性完整音频（base64）
- 调用方式：HTTP POST + JSON
- 请求头：
  - `Authorization: Bearer <API_KEY>`
  - `Content-Type: application/json`

请求体：

```json
{
  "text": "Hello from native batch endpoint.",
  "reference_audio_path": "/home/ubuntu/project/public_api_server/tts_server/assets/audio/en_3.wav"
}
```

成功响应（`data` 字段）：

- `mode`: 固定 `batch`
- `sample_rate`: 采样率
- `audio_base64`: WAV 音频的 base64
- `byte_length`: 原始音频字节长度
- `backend`: 当前推理后端

### 5.4 `POST /tts/stream`

- 功能：文本转语音，返回分片结果（base64 分片 + 聚合结果）
- 调用方式：HTTP POST + JSON
- 请求头：
  - `Authorization: Bearer <API_KEY>`
  - `Content-Type: application/json`

请求体：

```json
{
  "text": "Hello from native stream endpoint.",
  "reference_audio_path": "/home/ubuntu/project/public_api_server/tts_server/assets/audio/en_3.wav"
}
```

成功响应（`data` 字段）：

- `mode`: 固定 `stream`
- `sample_rate`: 采样率
- `chunks_base64`: PCM 分片数组（base64）
- `chunk_count`: 分片数量
- `aggregated_audio_base64`: 全量聚合音频（base64）
- `byte_length`: 聚合后字节长度
- `backend`: 当前推理后端

### 5.5 `GET /v1/models`

- 功能：返回 OpenAI 兼容模型列表
- 调用方式：HTTP GET
- 请求头：`Authorization: Bearer <API_KEY>`

示例响应（节选）：

```json
{
  "object": "list",
  "data": [
    {"id": "tts-1", "object": "model", "created": 1714300000, "owned_by": "local"},
    {"id": "tts-1-hd", "object": "model", "created": 1714300000, "owned_by": "local"},
    {"id": "moss-tts-nano-onnx", "object": "model", "created": 1714300000, "owned_by": "local"}
  ]
}
```

### 5.6 `GET /v1/audio/voices`

- 功能：返回可用音色列表（含别名映射和提示音频可用性）
- 调用方式：HTTP GET
- 请求头：`Authorization: Bearer <API_KEY>`

示例响应（节选）：

```json
{
  "object": "list",
  "data": [
    {
      "canonical": "Adam",
      "aliases": ["alloy", "onyx"],
      "prompt_audio_file": "en_4.wav",
      "prompt_audio_path": "/home/ubuntu/project/public_api_server/tts_server/assets/audio/en_4.wav",
      "prompt_audio_configured": true,
      "prompt_audio_exists": true
    }
  ]
}
```

### 5.7 `POST /v1/audio/speech`

- 功能：OpenAI 兼容 TTS 合成接口
- 调用方式：
  - `application/json`
  - 或 `multipart/form-data`（支持文件上传参考音频）
- 请求头：`Authorization: Bearer <API_KEY>`

JSON 请求体字段：

- `model` (string, 必填)
- `input` (string, 必填，非空)
- `voice` (string, 必填)
- `response_format` (string, 可选，默认 `wav`，仅支持 `wav`/`pcm`)
- `stream` (bool, 可选，默认 `false`)
- `reference_audio_url` (string, 可选，http/https)
- `reference_audio_path` (string, 可选，本地路径)

`multipart/form-data` 额外支持：

- `reference_audio_file` (file, 可选)

参考音频优先级（高到低）：

1. `reference_audio_file`
2. `reference_audio_url`
3. `reference_audio_path`
4. 内置 `voice`

请求示例（JSON 非流式）：

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello from non-stream API example.",
    "voice": "alloy",
    "response_format": "wav",
    "stream": false
  }' \
  --output /tmp/tts-example.wav
```

请求示例（multipart + 上传参考音频）：

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" \
  -H "Authorization: Bearer <API_KEY>" \
  -F "model=tts-1" \
  -F "input=Use uploaded prompt audio." \
  -F "voice=alloy" \
  -F "response_format=wav" \
  -F "stream=false" \
  -F "reference_audio_file=@./assets/audio/en_3.wav" \
  --output /tmp/tts-ref-upload.wav
```

响应行为：

- `stream=false`：返回完整音频字节流
  - `response_format=wav` -> `Content-Type: audio/wav`
  - `response_format=pcm` -> `Content-Type: audio/pcm`
- `stream=true`：返回分块 `StreamingResponse`
  - `Content-Type` 同上
  - 适合边下边播

## 6. 常见错误码

- `400 Bad Request`
  - 参数不合法（如 `response_format` 非 `wav|pcm`）
  - `model` 不支持
  - `input` 为空
  - 参考音频 URL 非 `http/https` 或文件超限（>8MB）
- `401 Unauthorized`
  - 缺少/错误 Bearer Token
- `404 Not Found`
  - `/tts/*` 模式下 `reference_audio_path` 文件不存在
- `500 Internal Server Error`
  - 运行时合成失败（例如 `/v1/audio/speech` 返回 `detail: speech synthesis failed`）

## 7. 对接建议

- 对接 OpenAI SDK 或外部系统时，优先使用 `POST /v1/audio/speech`。
- 若你需要统一 JSON 包裹便于日志采集/调试，可使用 `/tts/batch` 和 `/tts/stream`。
- 正式环境建议保持：
  - `AUTH_REQUIRED=true`
  - `DOCS_PUBLIC_IN_DEV=false`
