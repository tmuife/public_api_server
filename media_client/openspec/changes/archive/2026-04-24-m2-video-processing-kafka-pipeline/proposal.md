## Why

当前 `media_client` 的能力重点仍偏向诊断与基础连通性，和现在要落地的“视频处理最小业务闭环”不一致。为了尽快进入开发实施，需要先以 OpenSpec 明确一个更简单、可执行、以视频处理为中心的 API 与 Kafka 数据面方案。

## What Changes

- 架构与配置基线：
  - `main.py` 作为 FastAPI 唯一主入口。
  - 所有业务代码放在 `app/`，按 `routers/services/utils` 分层，避免过度设计。
  - 使用 `python-decouple` 读取配置，遵循“环境变量优先，其次 `.env`”。
- API 收敛（仅保留 3 个核心接口）：
  - `POST /videos/process-by-path`：输入本地视频路径。
  - `POST /videos/process-upload`：上传视频并写入配置目录后处理。
  - `POST /videos/compose`：输入完整 Kafka topic 名，直接读取该 topic 合成视频。
- 前处理流程标准化：
  - 为每次任务生成 `job_name`，规则为 `job_{utc毫秒时间戳}_{8位随机串}`。
  - 自动创建 Kafka topic：`{job_name}_input`。
  - 执行拆帧、抽取原音频、提取视频元数据（fps/bitrate/分辨率/总帧数）。
  - 使用 `AES-256-GCM` 对每帧加密（密钥来自 `.env`）。
  - 帧索引与加密后帧数据只通过 Kafka 传输，不使用第三方/对象存储。
- 后处理（合成）流程标准化：
  - `compose` 接口接收完整 topic（例如 `job_xxx_output`），直接消费该 topic。
  - 不依赖结束标志；基于前处理记录的总帧数判断是否收齐，配合超时机制。
  - 读取帧后按原 fps 与原码率、结合原音频合成新视频。
  - 返回合成结果文件路径。
- 存储边界：
  - 本地仅使用 `WORK_DIR` 管理作业目录与中间文件（帧、音频、元数据、输出）。
  - 禁止引入第三方文件存储或对象存储作为帧中转。
- 向后兼容影响（**BREAKING**）：
  - 现有以 doctor/诊断为中心的接口与调用路径不再是主要业务形态。

## Capabilities

### New Capabilities
- `video-preprocess-api-minimal`: 定义路径入参与上传入参两类前处理 API，以及统一作业初始化流程。
- `video-compose-api-topic-driven`: 定义基于完整 topic 输入的后处理合成 API 与返回结果路径语义。
- `video-workdir-job-lifecycle`: 定义 `WORK_DIR` 下作业目录组织、元数据持久化、临时文件生命周期。
- `video-kafka-job-topic-bootstrap`: 定义 `job_name` 生成、`{job_name}_input` 主题创建、主题命名约束。

### Modified Capabilities
- `fastapi-service-foundation`: 明确 `main.py` 作为主入口并承载视频处理 API 主路由。
- `client-kafka-secure-bootstrap`: 从“诊断场景配置”扩展为“业务作业场景配置”（含动态 topic 创建所需参数）。
- `encrypted-frame-contract-foundation`: 强化为 `AES-256-GCM` 帧加密契约，并明确“仅 Kafka 传输、不落第三方存储”。

## Impact

- 受影响代码：
  - `main.py`
  - `app/routers/*`（新增视频处理接口）
  - `app/services/*`（视频拆帧/加密/Kafka 生产消费/视频合成）
  - `app/utils/*`（作业命名、路径管理、加解密、FFmpeg 封装）
  - 配置模型与 `.env.example`
- 外部依赖与系统影响：
  - Kafka：需要支持动态 topic 创建与消息读写；`compose` 读取外部生产者写入的输出 topic。
  - 本地运行环境：需要可执行的 FFmpeg/FFprobe。
- 数据与安全影响：
  - AES-256-GCM 密钥仅来自环境配置，不进入日志和 Kafka header。
  - 帧数据只在本地临时文件与 Kafka 中流转，不引入对象存储。
