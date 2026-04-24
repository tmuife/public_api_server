## Context

当前 `m1-kafka-client-foundation` 已明确变更方向：`media_client` 必须从 CLI-first 重构为 FastAPI-first。Bearer 认证、`/health` 豁免、`/docs` 调试能力都应由 `media_client/main.py` 启动的 FastAPI 服务直接提供，而不是依赖外部服务。

同时，Kafka 仍是加密帧处理的核心数据面，M1 要求保留并强化 `doctor` 的真实读写探针能力（metadata + produce/consume）。

## Goals / Non-Goals

**Goals:**

1. 将 `main.py` 重构为 FastAPI 服务入口，并具备标准生命周期。
2. 建立统一 Bearer 鉴权基线：业务接口默认鉴权，`/health` 免鉴权，`/docs` 可访问且可授权。
3. 将 Kafka 配置与诊断能力服务化（可由 API 调用），不是只存在 CLI 命令路径。
4. 为 M2 保留帧处理接口扩展点（jobs/frames/status/result）。

**Non-Goals:**

1. 本阶段不实现完整视频业务流程（拆帧、合成、worker 业务算法）。
2. 本阶段不引入对象存储。
3. 本阶段不引入 OAuth2/OIDC 等复杂身份体系，仍用静态 Bearer token。

## Decisions

### Decision 1: 主入口改为 FastAPI app

- 选择：`main.py` 导出可被 `uvicorn main:app` 启动的 `app` 对象。
- 原因：
  - 与“通过 `/docs` 直接调试”目标一致。
  - 对外接口、鉴权、OpenAPI、诊断能力可统一管理。
- 备选：继续 CLI-only，再新建单独 API 项目。
- 未选原因：会造成双入口分裂，增加维护与联调成本。

### Decision 2: 鉴权中间件采用全局默认保护 + 明确豁免路径

- 选择：全局 Bearer 鉴权中间件（或依赖）默认作用于全部业务路由。
- 豁免路径：`/health`、`/docs`、`/openapi.json`、`/docs/oauth2-redirect`。
- token 来源：`MEDIA_API_KEY`。
- 原因：
  - 符合用户明确要求。
  - 行为可预测，减少“漏加鉴权”风险。
- 备选：逐路由加依赖。
- 未选原因：易漏配，难保障全局一致性。

### Decision 3: `/docs` 保持公开页面访问，接口调用必须 Bearer

- 选择：docs 页面公开可访问，OpenAPI 声明 Bearer scheme，调用受保护接口需 Authorize。
- 原因：
  - 提升调试效率。
  - 不牺牲接口安全性。
- 约束：生产环境建议叠加网络层白名单或网关限制。

### Decision 4: `doctor` 作为服务能力，执行 staged 检查 + 真实读写

- 选择：`doctor` 逻辑保留并服务化，可通过受保护 API 触发。
- 检查阶段：
  1. `config`
  2. `api_baseline`
  3. `kafka.metadata`
  4. `kafka.read_write`
- 真实读写 topic：`KAFKA_TOPIC_DOCTOR_RW`。
- 输出：统一机器可读 JSON。
- 备选：只做 metadata。
- 未选原因：无法验证真实可用性。

### Decision 5: Kafka 消息契约继续执行“敏感字段禁令”

- 选择：Kafka payload/header 中禁止 key/password/token 等敏感字段。
- 原因：
  - 降低泄露面。
  - 与现有设计文档保持一致。

## Risks / Trade-offs

- [FastAPI 改造会破坏既有 CLI 使用方式] -> 提供过渡方案（保留子命令或单独脚本）并在 README 标注迁移路径。
- [公开 docs 增加探测面] -> 接口仍 Bearer 强制，生产建议加网络访问控制。
- [doctor 真实读写依赖 Kafka ACL 完整] -> 部署文档中明确 `video.doctor.rw.v1` 读写与 metadata 权限。
- [单项目同时承载 API 与诊断逻辑可能复杂化] -> 按模块分层（auth/router/services/kafka/config），并补齐单测。

## Migration Plan

1. 将 `main.py` 转为 FastAPI 应用入口，保留最小兼容层（如需要）。
2. 引入全局 Bearer 鉴权与豁免路径机制。
3. 接入 OpenAPI Bearer 声明，验证 `/docs` 调试链路。
4. 将 doctor 逻辑收敛为服务层，并提供受保护接口。
5. 更新测试、文档与运维说明，完成 M1 验收。

## Open Questions（建议默认值）

1. CLI 是否完全移除？
   - 建议：M1 保留最小 CLI shim（仅提示或调用 API），M2 再决定是否彻底移除。

2. doctor API 使用 GET 还是 POST？
   - 建议：使用 `POST /v1/system/doctor`，便于扩展参数与审计。

3. docs 是否在生产环境完全公网开放？
   - 建议：功能上允许访问，但部署层默认加来源限制。
