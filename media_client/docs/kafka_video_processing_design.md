# Kafka + FastAPI 视频处理系统详细设计（`media_client`）

## 1. 文档信息

- 文档版本：`v2.0`
- 更新日期：`2026-04-23`
- 适用项目：`/home/ubuntu/project/public_api_server/media_client`
- 文档目标：将项目定位从“CLI 客户端”调整为“FastAPI 服务优先”，并定义基于 Kafka 的生产级视频帧处理系统设计。

---

## 2. 架构重定位

## 2.1 关键变化

本项目不再以 `main.py` CLI 作为主形态，改为 **FastAPI 服务主入口**。

- 旧思路：`media_client` 作为本地 CLI 工具执行 doctor/提交任务。
- 新思路：`media_client` 提供标准 HTTP API（含 `/docs`），用户可：
  1. 通过脚本调用 API。
  2. 通过 `/docs` 手动调试和触发流程。

## 2.2 统一目标

1. 提供可访问、可调试、可鉴权的 API 控制面。
2. 使用 Kafka 作为加密帧主通道（不引入对象存储）。
3. 支持生产级认证、可靠性和诊断能力。

---

## 3. 系统边界与角色

## 3.1 组件角色

1. `media_client`（FastAPI 服务）：
   - 鉴权、请求校验、任务接入、Kafka 生产/消费、状态查询、doctor。
2. `Worker Pool`（远端处理服务）：
   - 消费加密帧 -> 解密 -> 图像处理 -> 加密 -> 回写 Kafka。
3. `Producer/Consumer Client`（可选）：
   - 可是本地脚本，也可以直接在 `/docs` 页面调试 API。
4. `Kafka Cluster`：
   - 请求帧、结果帧、事件流与死信队列。

## 3.2 设计原则

1. API 是控制与接入标准面，Kafka 是内部异步数据面。
2. 所有敏感信息只在服务端环境变量中管理，不进入 Kafka。
3. 认证默认开启，只有明确豁免路径可匿名访问。

---

## 4. 总体架构图

```text
┌────────────────────── Producer (Script or /docs) ──────────────────────┐
│ 1) 调用 FastAPI 创建 job                                                 │
│ 2) 发送加密帧到 FastAPI                                                  │
│ 3) 查询处理状态/读取处理结果                                             │
└──────────────────────────────────────────────────────────────────────────┘
                                  │ HTTP(Bearer)
                                  ▼
┌──────────────────────── media_client FastAPI ───────────────────────────┐
│ A) Bearer 鉴权中间件                                                     │
│ B) 请求校验与契约校验                                                    │
│ C) Producer: 发布 FrameRequest 到 Kafka                                  │
│ D) Consumer/Query: 聚合 FrameResult 供客户端查询                         │
│ E) Doctor: Kafka metadata + 实际 produce/consume 探针                    │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                      ┌──────────────────────────┐
                      │      Kafka Cluster       │
                      │ request/result/event/dlq │
                      └──────────────────────────┘
                                  │
                                  ▼
┌────────────────────────── Worker Pool ───────────────────────────────────┐
│ consume request -> decrypt -> process -> encrypt -> publish result       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. API 认证与访问策略

## 5.1 认证模型

采用 Bearer Token 认证：

- Header：`Authorization: Bearer <token>`
- token 来源：服务端 `.env` 中 `MEDIA_API_KEY`

## 5.2 路由策略（强制）

1. 默认：所有对外业务接口都必须 Bearer。
2. 免鉴权：
   - `/health`
   - `/docs`
   - `/openapi.json`
   - `/docs/oauth2-redirect`
3. `/docs` 页面可打开，但调用受保护接口时必须先 Authorize。

## 5.3 OpenAPI 安全声明

FastAPI OpenAPI 必须包含 HTTP Bearer scheme，确保 Swagger UI 可直接录入 token 调试。

---

## 6. API 契约（M1~M2 目标）

## 6.1 M1 必须落地

1. `GET /health`（public）
2. `GET /docs`（public）
3. `GET /openapi.json`（public）
4. `POST /v1/system/doctor` 或 `GET /v1/system/doctor`（protected）
   - 执行 Kafka staged 检查（包含真实读写）

## 6.2 M2 建议落地

1. `POST /v1/jobs`：创建任务
2. `POST /v1/jobs/{job_id}/frames`：提交加密帧
3. `POST /v1/jobs/{job_id}/seal`：标记帧提交完成
4. `GET /v1/jobs/{job_id}/status`：查询任务状态
5. `GET /v1/jobs/{job_id}/results`：读取已处理帧（或分页读取）

---

## 7. Kafka 主题设计

| Topic | 方向 | Key | 说明 |
|---|---|---|---|
| `video.frame.request.v1` | API -> Worker | `job_id` | 待处理密文帧 |
| `video.frame.result.v1` | Worker -> API | `job_id` | 已处理密文帧 |
| `video.job.event.v1` | 双向 | `job_id` | 生命周期事件 |
| `video.frame.dlq.v1` | Worker -> Ops | `job_id` | 失败死信 |
| `video.doctor.rw.v1` | API self-check | `probe_id` | doctor 实际读写探针 |

---

## 8. 消息契约（核心字段）

## 8.1 FrameRequest

必须字段：

- `job_id`
- `frame_index`
- `encrypted_frame_bytes`
- `trace_id`
- `attempt`

可选分片字段：

- `chunked`, `chunk_index`, `chunk_count`

## 8.2 FrameResult

必须字段：

- `job_id`
- `frame_index`
- `status`
- `encrypted_frame_bytes`

建议字段：

- `error_code`, `error_message`, `process_latency_ms`, `worker_id`

## 8.3 敏感字段禁令（强制）

Kafka 消息体和 Header 严禁出现：

- 明文密钥
- 密钥密文（wrapped key）
- API token
- 密码/凭据

---

## 9. Doctor 设计（M1 必做）

## 9.1 检查阶段

1. `config`：环境配置与参数校验
2. `api_baseline`：`/health`、`/docs`、Bearer scheme、受保护路由校验
3. `kafka.metadata`：broker/topic 元数据可见性
4. `kafka.read_write`：真实 produce + consume 闭环

## 9.2 结果格式

统一机器可读 JSON：

- `ok`
- `summary`
- `stages[]`（每个 stage 包含 `status/error_code/hint/details`）

失败返回非零状态码。

---

## 10. 安全设计

## 10.1 传输与认证

1. Kafka 强制 `SASL_SSL`。
2. API 强制 Bearer（除豁免路径）。
3. broker ACL 最小权限。

## 10.2 密钥管理

1. 业务加密密钥仅存服务端环境变量。
2. 建议 `FRAME_MASTER_KEY_HEX` + `FRAME_KEY_VERSION`。
3. 建议按周期轮换并保留短暂兼容窗口。

## 10.3 日志安全

1. token/password/key 一律脱敏。
2. 禁止记录完整 `encrypted_frame_bytes`。
3. 禁止外部响应暴露内部栈详情。

---

## 11. 可靠性设计

1. 投递语义：`At-least-once` + 幂等。
2. 幂等键：`(job_id, frame_index[, chunk_index])`。
3. 重试：指数退避，超限进入 `DLQ`。
4. 断点恢复：通过 job 状态与消费位点恢复。

---

## 12. 性能与容量

1. 单消息大小受限，默认启用帧压缩。
2. 超大帧启用 chunk 分片。
3. API 层应限制 in-flight 请求数并做背压。
4. Kafka 参数需联调：
   - `message.max.bytes`
   - `max.request.size`
   - `max.partition.fetch.bytes`

---

## 13. 配置规范

## 13.1 API 配置

1. `MEDIA_API_KEY`
2. `MEDIA_API_HEALTH_PATH`（默认 `/health`）
3. `MEDIA_API_DOCS_PATH`（默认 `/docs`）
4. `MEDIA_API_OPENAPI_PATH`（默认 `/openapi.json`）
5. `MEDIA_API_DOCS_OAUTH2_REDIRECT_PATH`（默认 `/docs/oauth2-redirect`）

## 13.2 Kafka 配置

1. `KAFKA_BOOTSTRAP_SERVERS`
2. `KAFKA_SECURITY_PROTOCOL`
3. `KAFKA_SASL_MECHANISM`
4. `KAFKA_SASL_USERNAME`
5. `KAFKA_SASL_PASSWORD`
6. `KAFKA_SSL_CA_FILE`
7. `KAFKA_TOPIC_FRAME_REQUEST`
8. `KAFKA_TOPIC_FRAME_RESULT`
9. `KAFKA_TOPIC_JOB_EVENT`
10. `KAFKA_TOPIC_DLQ`
11. `KAFKA_TOPIC_DOCTOR_RW`

---

## 14. 分阶段里程碑

1. **M1（当前）**：FastAPI 化 + Bearer 基线 + doctor 真实读写
2. **M2**：jobs/frames API + Kafka request/result 闭环
3. **M3**：结果聚合与状态查询、DLQ 与重试策略
4. **M4**：压测、监控、告警、运维手册

---

## 15. 验收标准（M1）

1. `main.py` 可启动 FastAPI 服务。
2. `/health` 可匿名访问。
3. `/docs` 可打开并可使用 Bearer 调试受保护接口。
4. 未携带 Bearer 访问保护路由返回 401。
5. doctor 能输出 Kafka metadata + 实际读写结果。
6. 所有敏感信息在日志和响应中脱敏。

---

## 16. 风险与缓解

1. 风险：`/docs` 公开导致探测面扩大
   - 缓解：接口仍 Bearer 强制，生产建议加网络白名单。
2. 风险：Kafka ACL 不完整导致 doctor 假失败
   - 缓解：明确 `video.doctor.rw.v1` 的读写授权。
3. 风险：消息过大导致 broker 压力
   - 缓解：压缩 + 分片 + 限流 + 参数联调。

---

本文作为后续“FastAPI 代码改造”与“接口实现”阶段的设计基线。
