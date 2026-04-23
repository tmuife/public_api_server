## Why

当 runtime 与 OpenAI 兼容 API 都迁移到当前项目后，需要一套“与参考目录解耦”的验收体系。
该体系必须验证：即使删除 `MOSS-TTS-Nano-main` 与 `Kokoro-FastAPI-master`，当前项目仍能独立通过 stream/batch 核心场景。

## What Changes

- 在当前项目新增 ONNX API 验收测试，覆盖 stream、batch、异常路径、最小依赖启动。
- 提供本地验证脚本（curl / OpenAI SDK）用于快速冒烟。
- 更新当前项目 README 与运行说明，明确依赖矩阵、支持格式、已知限制。
- 将“删除参考目录后验收通过”纳入发布前必检项。

## Capabilities

### New Capabilities
- `onnx-api-acceptance-validation`: 当前项目内可重复执行的 ONNX OpenAI API 自动化 + 手工验收机制。

### Modified Capabilities
- None.

## Impact

- Affected code (current project only):
  - 测试目录（新增 API 集成测试）
  - 文档（`README.md` 与运行指南）
  - 验收脚本目录（如 `scripts/`）
- Delivery process:
  - 发布门禁增加“参考目录删除后仍通过”校验

## Acceptance Criteria

- 至少一组自动化测试覆盖：stream 成功、batch 成功、invalid model、empty input。
- 在无 `torch/torchaudio` 且已删除参考目录的环境中，服务可启动并完成最小调用验证。
- README 明确写出当前项目启动命令、请求示例、返回格式、非目标与已知限制。
- 任一验收失败可定位具体失败场景与错误信息。
