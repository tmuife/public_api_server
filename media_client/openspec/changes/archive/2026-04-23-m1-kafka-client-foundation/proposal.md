## Why

当前 change 的实现假设偏向 CLI-first，与最新目标出现偏差：你希望 `media_client/main.py` 本身是 FastAPI 服务，并把 Bearer 认证、`/health` 豁免、`/docs` 调试能力落实在这个服务里。为了让后续代码改造方向一致，必须先修正提案基线。

## What Changes

- 架构方向调整（**BREAKING**）：
  - 将 `media_client` 的主形态从“CLI 工具优先”调整为“FastAPI 服务优先”。
  - `main.py` 应提供可启动的 FastAPI app（CLI 可作为附属工具保留或拆分）。
- 认证与路由策略：
  - 所有对外业务接口默认要求 `Authorization: Bearer <token>`。
  - `/health` 保持免鉴权。
  - `/docs`、`/openapi.json`、`/docs/oauth2-redirect` 可访问；在 docs 中使用 Bearer 登录调试受保护接口。
  - Bearer token 来源为 `.env` 中 `MEDIA_API_KEY`。
- Kafka 与诊断基线：
  - 保持 Kafka 作为核心消息通道（含认证配置）。
  - `doctor` 必须做真实读写（produce + consume），不仅仅是 metadata 检查。
- 边界约束：
  - 密钥、密码、token 只允许出现在环境变量，不进入 Kafka 消息体或 Header。

## Capabilities

### New Capabilities
- `fastapi-service-foundation`: 将 `media_client` 建立为可运行 FastAPI 服务（应用入口、路由组织、生命周期）。
- `api-bearer-auth-baseline`: Bearer 强制鉴权 + `/health` 豁免 + docs 可访问与可授权调试。
- `kafka-doctor-rw-baseline`: 提供 Kafka staged 诊断与真实读写探针能力。
- `encrypted-frame-contract-foundation`: 约束 Kafka 帧消息最小字段与敏感字段禁令。

### Modified Capabilities
- `client-kafka-secure-bootstrap`: 从 CLI-only 配置加载，升级为 FastAPI 服务内统一配置与校验能力。
- `client-kafka-doctor-rw`: 从 CLI 命令入口，升级为服务侧可调用能力（可对外暴露受保护诊断接口）。

## Impact

- 受影响代码：
  - `main.py`（从 CLI 入口向 FastAPI 入口重构）
  - 鉴权中间件/依赖、OpenAPI 安全声明、路由豁免机制
  - Kafka 适配与 doctor 诊断模块的服务化接入
  - `.env.example` 与 README（服务启动、Bearer、docs、doctor 说明）
- 受影响系统：
  - 调用方从“直接 CLI”转向“调用 HTTP API 或 `/docs`”。
  - Kafka ACL 需覆盖 doctor RW topic 的读写权限。
- 迁移影响：
  - 现有 CLI 使用方式将发生变化，需提供向后兼容策略或迁移说明。
