## Context

现有仓库中的参考目录仅用于对照实现，不应成为当前项目 API 的运行时依赖。
OpenAI 兼容层需要直接挂载在当前项目服务入口上，并调用本地 ONNX backend。

## Goals / Non-Goals

**Goals:**
- 在当前项目提供最小可用 OpenAI 兼容 TTS API（`/v1/audio/speech` + `/v1/models`）。
- stream 与 batch 共用同一后端生成链路，降低维护成本。
- 参考目录删除后接口仍然可运行。

**Non-Goals:**
- 不在本 change 中实现 OpenAI 全量音频接口。
- 不在本 change 中实现鉴权、计费与多租户。
- 不在本 change 中引入 PyTorch 依赖。

## Decisions

- 决策 1：采用“Router -> Service Facade -> Local ONNX Backend”三层结构，全部放在当前项目。
  - Rationale: 清晰分离协议适配与推理逻辑，并满足独立可维护性。
- 决策 2：batch 通过聚合 stream 生成结果获得，避免双实现分叉。
  - Rationale: 降低行为不一致和回归风险。
- 决策 3：`response_format` 首版严格限制为 `wav/pcm`，不支持格式统一返回 `HTTP 400`。
  - Rationale: 先保证兼容可用与依赖最小化，并用明确的参数错误语义降低客户端排障成本。
- 决策 4：`voice` 首版同时支持内置 voice 与参考音频（上传/引用）两种路径；当两者同时提供时，优先使用参考音频。
  - Rationale: 同时覆盖 OpenAI SDK 常见用法（内置 voice）与本地业务常见用法（voice clone）。

## Risks / Trade-offs

- [Risk] 部分客户端默认 `mp3`，首版仅 `wav/pcm` 需额外适配。
  - Mitigation: 文档明确声明并在后续 change 扩展格式。
- [Risk] 参考音频上传/引用会引入输入大小、来源可达性与安全性边界。
  - Mitigation: 对上传大小、时长和采样率做上限校验；对 URL 引用加超时和白名单策略；路径引用仅允许受控目录。
- [Risk] 反向代理缓冲导致流式体验退化。
  - Mitigation: 设置禁缓冲 header，并在文档中给出代理配置建议。
- [Risk] 参考实现迁移不完整导致边缘参数不兼容。
  - Mitigation: 以真实 SDK 调用场景做集成验收。

## Migration Plan

- 第一步：在当前项目新增 `/v1` 路由与请求 schema。
- 第二步：路由接入本地 ONNX backend，打通 stream/batch。
- 第三步：执行“删除参考目录后”兼容验收。
- 回滚策略：临时下线 `/v1` 路由，保留原有本地接口。

## Open Questions (Resolved on 2026-04-22)

- `voice`：首版支持内置 voice，也支持上传/引用参考音频；若同时提供，参考音频优先。
- `response_format`：首版严格限制为 `wav/pcm`，不支持格式返回 `HTTP 400`。
