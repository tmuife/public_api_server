## Context

本变更目标是在 `media_server` 中定义一个最小可实施的服务端处理面，用来承接 `media_client` 已完成的预处理链路。当前已确认的外部事实如下：

- `media_client` 预处理后会将帧写入 `{job_name}_input`。
- `media_client` 合成阶段会从 `{job_name}_output` 读取结果帧。
- 输入帧消息字段为：
  - `job_name`
  - `frame_index`
  - `nonce_b64`
  - `ciphertext_b64`
  - `tag_b64`
  - `content_type`
- 输出帧消息字段为：
  - `job_name`
  - `frame_index`
  - `status`
  - `nonce_b64`
  - `ciphertext_b64`
  - `tag_b64`
  - 可选 `error_code`（当 `status=error` 时必填）
- topic 命名只能使用 `^[a-zA-Z0-9._-]+$`
- `job_name` 由 topic 推导，输入 topic 为 `{job_name}_input`，输出 topic 为 `{job_name}_output`
- 当前 `media_client` 的输入消息里不包含 `frame_count`，也没有 EOF/seal 消息

因此，`media_server` 的设计重点不是视频编排，而是：

1. 用最小 HTTP 接口异步启动一次 topic 处理会话。
2. 严格兼容现有消息契约。
3. 在没有结束标志的前提下，定义可预测的消费收敛方式。

## Goals / Non-Goals

**Goals:**

1. 提供一个标准 FastAPI 接口，接收 `topic` 并异步触发一次端到端处理。
2. 保持与 `media_client` 的 Kafka topic、job_name 推导、AES-256-GCM 加密字段完全兼容。
3. 明确错误分类和服务端响应格式，便于联调。
4. 将“具体图像处理逻辑”隔离成单独处理器，先以占位实现打通链路。
5. 避免在 `media_server` 引入不必要状态，维持简单、可维护的实现。

**Non-Goals:**

1. 不实现视频预处理、视频合成、workspace 或 manifest 管理。
2. 不新增任务列表、任务状态查询、取消、重试等管理类接口。
3. 不引入数据库、对象存储、消息编排器或后台任务系统。
4. 不在本期设计多 topic 批处理或长期驻留型 consumer daemon。

## Architecture Overview

```text
Caller
  |
  | POST /videos/process-topic
  | { "topic": "{job_name}_input" }
  v
media_server FastAPI
  |
  | validate request + auth
  | enqueue background task
  | return accepted response
  v
Background Task Runner
  |
  | derive job_name / output_topic
  | build deterministic consumer group.id
  v
Kafka consume loop
  |
  | validate frame request contract
  | decrypt AES-256-GCM
  | process frame (placeholder)
  | encrypt AES-256-GCM
  | publish frame result
  v
{job_name}_output
```

服务分层保持最小：

- `main.py`: 应用启动、生命周期、异常处理、健康检查
- `app/routers/videos.py`: HTTP 路由、请求模型、调用 service
- `app/services/topic_process_service.py`: 异步任务提交与主处理流程
- `app/services/kafka_service.py`: Kafka topic ensure / consumer / producer 封装
- `app/utils/contracts.py`: 契约校验
- `app/utils/crypto.py`: AES-256-GCM 加解密
- `app/utils/job.py`: topic/job_name 规则
- `app/utils/runtime.py`: 进程内任务注册与运行摘要（仅用于本实例去重和诊断）

## Decisions

### Decision 1: 只保留一个异步触发业务接口

- 选择：
  - `POST /videos/process-topic`
  - 请求体：`{ "topic": "job_xxx_input" }`
  - 接口只负责校验参数、提交后台处理任务，并尽快返回 `202 Accepted`
  - 实际 topic 消费、处理、回写由后台任务执行
- 原因：
  - 处理图片时很可能需要线程池或其他并行处理方式
  - 避免长视频任务占用单个 HTTP 请求生命周期
  - 与“目前只需要一个接口”的目标一致
  - 不引入额外任务状态管理
- 备选：
  - 同步阻塞直到处理完成
- 未选原因：
  - 对长视频和多线程处理场景不友好，请求超时风险更高

### Decision 2: 服务端处理范围是“topic 级无状态 worker”

- 选择：
  - `media_server` 不维护作业目录
  - 不读取 `media_client` 的 manifest
  - 不感知音频、fps、码率等视频合成元数据
  - 仅负责帧级解密、处理、回写
- 原因：
  - 服务职责边界清晰，不与 `media_client` 重叠
  - 当前需求不要求持久化本地资产
  - 空项目更适合从无状态处理面起步
- 备选：
  - 在服务端也持有 job workspace 和元数据
- 未选原因：
  - 会复制 `media_client` 的职责并增加实现复杂度

### Decision 3: topic 输入必须是 `_input`，输出总是 `_output`

- 选择：
  - API 入参必须传完整输入 topic
  - topic 必须满足字符规则且以 `_input` 结尾
  - `job_name = topic[:-len("_input")]`
  - 输出 topic 固定推导为 `{job_name}_output`
- 原因：
  - 防止用户误把 output topic 作为输入
  - 与 client 既有命名保持一致
  - 服务内部无二义性
- 备选：
  - 接收任意 topic 后自动猜测其用途
- 未选原因：
  - 容易误用，且错误不易排查

### Decision 4: 完全复用 client 的 AES-256-GCM 与契约定义

- 选择：
  - 输入消息校验按 frame request contract 执行
  - 输出消息校验按 frame result contract 执行
  - 使用同一套 `nonce_b64/ciphertext_b64/tag_b64` 字段
  - 服务端和客户端共享相同的 32-byte AES key 来源
- 原因：
  - 当前链路兼容性是最高优先级
  - 降低双端协议漂移风险
- 备选：
  - 服务端引入独立协议或二次封装结构
- 未选原因：
  - 破坏 `media_client` 已实现的 compose 能力

### Decision 5: 失败帧也必须写回加密结果消息

- 选择：
  - 当单帧处理失败时，仍发布一条 `status=error` 的 result 消息
  - 该消息仍必须包含 `nonce_b64`、`ciphertext_b64`、`tag_b64`
  - `error_code` 使用机器可读短码，例如：
    - `FRAME_DECRYPT_FAILED`
    - `FRAME_PROCESS_FAILED`
    - `FRAME_ENCRYPT_FAILED`
- 原因：
  - `media_client` 当前对 result contract 的校验要求错误帧也带加密字段
  - 可以让 compose 侧获得稳定、可诊断的失败语义
- 备选：
  - 处理失败时只记录日志，不发消息
  - 只发 `error_code`，不带加密字段
- 未选原因：
  - 会导致 compose 侧无法按现有契约工作，或直接判定消息非法

### Decision 6: 用“首帧等待 + 空闲超时 + 总时长上限”作为收敛策略

- 选择：
  - `KAFKA_WAIT_FIRST_FRAME_SECONDS`：等待首帧的最大时长
  - `KAFKA_IDLE_TIMEOUT_SECONDS`：收到至少一帧后，无新帧到达的最大空闲时长
  - `KAFKA_MAX_PROCESS_SECONDS`：整次 topic 处理的总上限
  - 满足以下任一条件即结束本次会话：
    1. 超过首帧等待时间且仍无有效消息
    2. 已处理至少一帧且空闲超时
    3. 达到总时长上限
- 原因：
  - 当前输入 topic 不含总帧数，也没有结束标志
  - 这是在不修改 client 契约前提下最可实施的办法
- 备选：
  - 要求 client 额外发送 EOF/seal 消息
  - 服务端读取 `media_client` 本地 manifest
- 未选原因：
  - 第一种需要修改 client 协议
  - 第二种会让两个子项目耦合到同一工作目录

### Decision 7: 单次消费语义以固定 Kafka `group.id` 为主

- 选择：
  - 为同一输入 topic 构造确定性的 Kafka `group.id`
  - 同一 topic 的后台任务在多实例场景下使用同一个 `group.id`
  - 依赖 Kafka consumer group 分配与 offset commit，保证同一消息在该 group 内只被一个 consumer 处理
  - 进程内任务注册表只做 best-effort 去重，避免同一实例重复启动后台任务
- 原因：
  - 不需要额外引入外部锁或调度层
  - 更符合 Kafka 原生消费模型
  - 对未来多实例部署更自然
- 备选：
  - 使用 Redis/DB 做分布式锁
  - 强制单实例本地锁
- 未选原因：
  - 第一种增加额外组件复杂度
  - 第二种无法覆盖多实例部署场景

### Decision 8: 处理逻辑抽象成独立处理器，默认 passthrough

- 选择：
  - 定义 `FrameProcessor.process(frame_bytes, content_type) -> bytes`
  - 当前默认实现直接返回原始 `frame_bytes`
  - 后续替换真实算法时，不修改 API 与 Kafka 契约
- 原因：
  - 现在“处理过程留空”，但设计上要预留稳定扩展点
  - 能把链路逻辑与图像处理逻辑解耦
- 备选：
  - 把处理逻辑直接写进主 service
- 未选原因：
  - 后续接入模型或图像算子时会增加重构成本

## Detailed Flow

### Request Flow

1. 调用方请求 `POST /videos/process-topic`
2. Bearer 中间件校验 token
3. Router 校验请求体存在非空 `topic`
4. Service 执行 topic 规则校验：
   - 非空
   - 只包含允许字符
   - 必须以 `_input` 结尾
5. 从 input topic 推导：
   - `job_name`
   - `output_topic = {job_name}_output`
6. 构造该 topic 对应的确定性 `group.id`
7. 将后台处理任务注册到进程内任务表（若本实例已存在同 topic 任务，则直接复用）
8. 立即返回受理响应
9. 后台任务创建或确认输出 topic 存在
10. 启动 Kafka consume loop
11. 对每一条消息执行：
   - JSON 解析
   - request contract 校验
   - 若 `job_name` 不匹配则忽略
   - 解密 frame
   - 调用占位处理器
   - 重加密
   - 组装 `status=ok` result 消息
   - 发布到 output topic
12. 若某帧在处理过程中失败：
   - 组装 `status=error` 的 result 消息
   - 附带 `error_code`
   - 仍发布到 output topic
13. 每条消息在成功发布结果后提交对应 offset
14. 达到收敛条件后结束消费并清理本地任务注册

### Accepted Response

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "job_name": "job_1776945123456_ab12cd34",
    "input_topic": "job_1776945123456_ab12cd34_input",
    "output_topic": "job_1776945123456_ab12cd34_output",
    "group_id": "media-server-job_1776945123456_ab12cd34",
    "dispatch_mode": "async",
    "status": "accepted"
  }
}
```

HTTP status 建议为 `202 Accepted`。

### Error Semantics

请求受理阶段的 HTTP 错误：

- `400`
  - topic 为空
  - topic 字符非法
  - topic 不是 `_input`
- `500`
  - Kafka 依赖缺失
  - Kafka client 初始化失败
  - 后台任务提交前的未预期运行时异常

后台执行阶段的错误不再回到已返回的 HTTP 请求，而是通过以下方式暴露：

- 写入 `status=error` 的 output topic result 消息
- 记录服务端结构化日志
- 在进程内任务摘要中标记失败原因（若实现该诊断能力）

## Config Design

服务端配置保持与 client 同风格，最小必需项如下：

- `MEDIA_API_ACCESS_TOKEN`
- `AES_256_GCM_KEY_BASE64`
  或：
- `AES_256_GCM_PASSPHRASE`
- `AES_256_GCM_KDF_SALT_BASE64`
- `AES_256_GCM_KDF_ITERATIONS`
- `KAFKA_BOOTSTRAP_SERVERS`
- `KAFKA_SECURITY_PROTOCOL`
- `KAFKA_SASL_MECHANISM`
- `KAFKA_SASL_USERNAME`
- `KAFKA_SASL_PASSWORD`
- `KAFKA_SSL_CA_FILE`
- `KAFKA_PRODUCE_TIMEOUT_SECONDS`
- `KAFKA_WAIT_FIRST_FRAME_SECONDS`
- `KAFKA_IDLE_TIMEOUT_SECONDS`
- `KAFKA_MAX_PROCESS_SECONDS`
- `KAFKA_TOPIC_PARTITIONS`
- `KAFKA_TOPIC_REPLICATION_FACTOR`

额外建议：

- `MEDIA_API_HOST` 默认 `0.0.0.0`
- `MEDIA_API_PORT` 默认 `8000`

## Risks / Trade-offs

- [基于空闲超时收敛可能提前结束] -> 如果生产端长时间停顿，服务端可能过早结束；当前通过首帧等待、空闲超时、总时长三个参数尽量降低误判。
- [重复触发可能启动多个后台任务] -> 正确性依赖固定 `group.id` 和 offset commit；同实例内再用任务注册表做 best-effort 去重，降低资源浪费。
- [错误帧也需要加密回写] -> 这会让错误路径比普通 HTTP 错误复杂，但这是兼容现有 compose 校验的必要代价。
- [Kafka group 只能保证 group 内单次分配] -> 若部署策略或 `group.id` 生成规则漂移，仍可能出现重复处理，因此需要把 `group.id` 规则固化到实现与文档中。
- [异步接口缺少状态查询面] -> 本期接口只负责受理，不返回最终处理结果；调用方需要通过既有业务编排或下游消费时机来衔接。

## Migration Plan

1. 在 `media_server` 下建立 `openspec/changes/<change-name>/`。
2. 根据本 proposal/design 补 specs delta。
3. 搭建 FastAPI 基础骨架与鉴权中间件。
4. 迁移并裁剪与 client 对齐的 `config/contracts/crypto/job` 基础能力。
5. 实现 Kafka 消费、处理、回写链路、后台任务调度与固定 `group.id` 规则。
6. 用 passthrough processor 打通完整流程。
7. 补测试与 `README.md`。
