## Context

当前仓库包含 `MOSS-TTS-Nano-main` 与 `Kokoro-FastAPI-master` 两个参考目录，但目标实现必须落在当前项目自身代码中。
因此需要将 ONNX 推理所需逻辑抽取到本地模块，并切断任何对参考目录与 PyTorch 栈的运行时依赖。

## Goals / Non-Goals

**Goals:**
- 在当前项目内构建完全独立的 ONNX runtime 路径。
- 保持 voice clone、streaming decode 等关键能力。
- 确保参考目录删除后不影响运行。

**Non-Goals:**
- 不在本 change 中完成 OpenAI 兼容 API 全量实现。
- 不在本 change 中保留 legacy demo UI 的全部特性。
- 不要求与参考实现逐样本一致，只要求可用与稳定。

## Decisions

- 决策 1：本地模块命名统一为 `app/*` 分层目录（`app/runtime`、`app/services`、`app/utils`），不直接引用参考目录模块。
  - Rationale: 参考目录将被删除，必须避免耦合。
  - Alternative considered: 继续 import 参考目录并后续替换。风险高，易遗漏。
- 决策 2：参考音频 I/O 与重采样采用轻量非 torch 方案，且重采样优先库为 `soxr`。
  - Rationale: 去除 `torchaudio` 是 torchless 的必要条件。
- 决策 3：先完成 runtime 独立，再推进 API 兼容层。
  - Rationale: 先稳住基础依赖边界，降低后续接口改造风险。

## Risks / Trade-offs

- [Risk] 从参考目录抽取后，行为与原实现存在细微偏差。
  - Mitigation: 保留关键路径回归样例（短文本 + 固定参考音频）。
- [Risk] 抽取过程中遗漏隐式依赖。
  - Mitigation: 增加“删除参考目录后冒烟测试”作为强制验收。
- [Risk] 新音频依赖在个别平台安装成本较高。
  - Mitigation: 优先选择跨平台 wheel 友好依赖并固定版本。

## Migration Plan

- 第一步：在当前项目新增本地 ONNX runtime 与音频处理模块，完成代码抽取。
- 第二步：改造服务入口仅引用本地模块，移除对参考目录 import。
- 第三步：执行“删除参考目录后的启动与推理验收”。
- 回滚策略：保留抽取前分支，必要时短期回滚到旧入口进行排障。

## Open Questions

- None.
- 已确认：本地模块命名统一为 `app/*` 目录。
- 已确认：音频重采样优先使用 `soxr`。
