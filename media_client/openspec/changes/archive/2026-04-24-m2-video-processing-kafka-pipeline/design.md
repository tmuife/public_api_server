## Context

本变更用于把 `media_client` 收敛为一个“可直接实施”的 FastAPI 视频处理服务，避免继续围绕诊断接口扩展复杂度。已确认的约束如下：

- `main.py` 是主入口。
- 配置读取采用 `python-decouple`，优先系统环境变量，其次 `.env`。
- 代码全部放在 `app/`，允许 `routers/services/utils` 分层。
- 仅保留 3 个业务接口：路径处理、文件上传处理、后期合成。
- 后端至少依赖 Kafka；工作目录由 `WORK_DIR` 控制。
- 前处理必须：拆帧 -> AES-256-GCM 加密 -> 发送 Kafka。
- 合成必须：读取用户提供的完整 topic（通常为 `*_output`）-> 收齐帧 -> 结合原音频按原参数合成。
- 帧数据只通过 Kafka，不使用第三方/对象存储。
- 合成返回文件路径，不返回下载流。

## Goals / Non-Goals

**Goals:**

1. 建立一个最小但完整的视频处理闭环 API（3 个接口）。
2. 提供明确的作业命名、topic 命名和工作目录约束。
3. 固化 AES-256-GCM 帧级加密与 Kafka 传输契约。
4. 在无结束标志消息的前提下，提供可预测的合成收帧逻辑。
5. 输出可直接进入实现阶段的技术方案。

**Non-Goals:**

1. 不实现对象存储、第三方文件系统或 CDN 集成。
2. 不引入复杂任务编排系统（如 Celery/Argo）。
3. 不实现多租户配额、审计平台、可视化后台。
4. 不在本阶段设计额外业务接口（如任务列表、取消、重试管理）。

## Decisions

### Decision 1: 目录与模块结构采用最小分层

- 选择：
  - `main.py` 仅负责创建 FastAPI app 与挂载路由。
  - 业务代码放入 `app/`：
    - `app/routers/videos.py`
    - `app/services/video_process_service.py`
    - `app/services/video_compose_service.py`
    - `app/services/kafka_service.py`
    - `app/utils/crypto.py`, `app/utils/ffmpeg.py`, `app/utils/job.py`, `app/utils/pathing.py`
- 原因：满足“结构清晰 + 不复杂”的要求，便于快速开发与维护。
- 备选：引入更多层（repository/domain/application）。
- 未选原因：当前规模会过度设计。

### Decision 2: 配置体系统一使用 decouple 并定义必需环境变量

- 选择：通过 `python-decouple` 统一读取配置，关键变量包括：
  - `WORK_DIR`
  - `UPLOAD_DIR`（可选，默认 `WORK_DIR/uploads`）
  - `AES_256_GCM_KEY_BASE64`（32 字节 key 的 base64）
  - `KAFKA_BOOTSTRAP_SERVERS`
  - `KAFKA_SECURITY_PROTOCOL` / SASL 相关项（按部署需求）
  - `KAFKA_PRODUCE_TIMEOUT_SECONDS`
  - `KAFKA_CONSUME_TIMEOUT_SECONDS`（compose 收帧超时）
- 原因：配置行为可预测，符合“环境变量优先，再 `.env`”。
- 备选：手写配置加载。
- 未选原因：会重复造轮子且易遗漏类型转换。

### Decision 3: API 只保留三条并使用统一响应模型

- 选择：
  - `POST /videos/process-by-path`
  - `POST /videos/process-upload`
  - `POST /videos/compose`
  - 成功响应统一为 `{ code: 0, message: "success", data: ... }`
- 原因：接口面最小，满足现阶段目标；统一响应便于前端与调用方集成。
- 备选：拆出更多查询型接口（任务详情、状态轮询）。
- 未选原因：超出当前最小闭环需求。

### Decision 4: job/topic 规则固定，保证可追踪与可推导

- 选择：
  - `job_name = job_{utc毫秒时间戳}_{8位随机串}`
  - 前处理输入 topic：`{job_name}_input`
  - compose 入参是完整 topic 名，服务端不做强制重命名。
- 原因：唯一性、可读性、实现成本低。
- 备选：UUID-only 或数据库自增 ID。
- 未选原因：UUID 可读性较差；数据库 ID 引入额外状态依赖。

### Decision 5: 前处理采用“本地预处理 + Kafka 传输”

- 选择：
  1. 校验输入视频（路径或上传文件）并放入 `WORK_DIR/jobs/{job_name}/source/`。
  2. 用 FFmpeg/FFprobe 生成：
     - 帧序列（临时目录）
     - 原音频文件
     - 视频元数据（fps/bitrate/frame_count/codec/resolution）
  3. 对每帧执行 AES-256-GCM 加密并发送到 `{job_name}_input`。
  4. 将作业清单保存为 `WORK_DIR/jobs/{job_name}/manifest.json`。
- 原因：不依赖外部存储，同时为 compose 阶段提供稳定元数据。
- 备选：边拆帧边发 Kafka（无本地清单）。
- 未选原因：compose 需要稳定元数据（总帧数/原音频/码率），纯流式实现复杂度更高。

### Decision 6: 帧消息契约使用 AES-256-GCM 必需字段

- 选择：Kafka 帧消息至少包含：
  - `job_name`
  - `frame_index`（0-based）
  - `nonce_b64`（12-byte）
  - `ciphertext_b64`
  - `tag_b64`
  - `content_type`（如 `image/jpeg`）
- 原因：满足可解密和可重建所需最小信息。
- 备选：仅发送合并后的 `encrypted_frame_bytes`。
- 未选原因：解密需要 nonce/tag，拆字段更明确且可验证。

### Decision 7: compose 在“无结束标志”下按总帧数收敛

- 选择：
  - `compose` 接口接收完整 topic，并根据该 topic 解析/定位 `job_name` 对应 manifest。
  - 消费 topic 时按 `frame_index` 去重并累计，直到达到 `manifest.frame_count`。
  - 达不到则在 `KAFKA_CONSUME_TIMEOUT_SECONDS` 超时失败返回。
- 原因：你明确不需要结束标志，最稳妥做法是用已知总帧数判断完成。
- 备选：依赖外部发送 EOF 消息。
- 未选原因：已被需求排除。

### Decision 8: 合成参数以原视频元数据为准

- 选择：
  - 先将输出 topic 中帧按索引排序并解密到本地目录。
  - 使用 manifest 中的原 fps/bitrate 与抽取的原音频进行封装。
  - 输出到 `WORK_DIR/jobs/{job_name}/output/final.mp4`。
  - API 返回输出文件绝对路径。
- 原因：满足“按原码率 + 原音频合成”的确定性要求。
- 备选：固定编码参数。
- 未选原因：会偏离原视频质量与用户预期。

## Risks / Trade-offs

- [Kafka 单条消息体积可能过大] -> 约束帧编码质量与分辨率；必要时配置 broker/client `max.message.bytes` 并在服务启动校验。
- [无结束标志导致等待时间不可控] -> 使用总帧数+超时双条件；超时返回明确缺失帧统计。
- [AES key 配置错误导致全链路失败] -> 启动时校验 key 长度必须为 32 字节并快速失败。
- [本地 `WORK_DIR` 可能膨胀] -> 增加作业级清理策略（按成功/失败与 TTL 清理中间帧）。
- [外部输出 topic 消息格式漂移] -> 在 compose 阶段做严格 schema 校验并输出可定位错误信息。

## Migration Plan

1. 创建并确认本次 OpenSpec proposal/design（当前步骤）。
2. 按 proposal 产出对应 specs delta（新增能力与修改能力逐项落地）。
3. 实施代码重构：`main.py` + `app/routers/services/utils`。
4. 增加关键测试：
   - 配置加载优先级
   - job/topic 规则
   - AES-256-GCM 加解密
   - process/compose 主流程
5. 更新 `README.md` 与 `.env.example`，补齐运行与调用示例。

## Open Questions

1. 外部写入 `*_output` topic 的帧编码格式是否固定为 JPEG，还是允许 PNG/WebP（当前建议：允许，但必须在消息里声明 `content_type`）。
2. `compose` 入参仅有 `topic` 时，`job_name` 解析规则是否固定为去除 `_output` 后缀（当前建议：是；若无后缀则按原值尝试匹配作业目录）。
3. 上传文件清理策略是否需要“合成完成后立即删除原上传文件”（当前建议：默认保留到作业 TTL 到期再清理，便于排障）。
