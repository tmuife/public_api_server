## Why

当前目标是在**当前项目目录**内落地 ONNX TTS 服务，而不是继续依赖 `MOSS-TTS-Nano-main` 或 `Kokoro-FastAPI-master` 的代码树。
这两个目录仅用于实现前的对照与抽取，最终会被删除，因此必须先完成本项目内的 ONNX runtime 去 torch 化与独立化。

## What Changes

- 在当前项目中新建本地 ONNX runtime 模块与音频预处理模块，按需抽取参考实现。
- 本地模块命名统一收敛到 `app/*` 目录（如 `app/runtime/*`、`app/services/*`、`app/utils/*`）。
- 彻底移除运行链路中对 `torch/torchaudio` 的依赖，采用轻量音频库完成读取与重采样。
- 音频重采样优先使用 `soxr` 实现。
- 服务入口改为仅依赖当前项目本地模块，不再 `import` 参考目录下任何模块。
- 建立依赖边界约束，确保参考目录删除后服务仍可启动与推理。

## Capabilities

### New Capabilities
- `onnx-runtime-torchless`: 在当前项目内提供纯 ONNX 的推理运行时能力，允许无 PyTorch 环境启动与推理。

### Modified Capabilities
- None.

## Impact

- Affected code (current project only):
  - `main.py`（或重构后的本地入口）
  - 新增本地模块：`app/runtime/*`、`app/services/*`、`app/utils/*`（以最终实现为准）
- Dependencies:
  - 移除运行时对 `torch/torchaudio` 的必要性
  - 引入轻量音频 I/O 与重采样依赖
- Repository policy:
  - `MOSS-TTS-Nano-main` 与 `Kokoro-FastAPI-master` 仅作参考，不作为运行时依赖

## Acceptance Criteria

- 在仅安装当前项目依赖且不安装 `torch/torchaudio` 的环境中，服务可成功启动。
- 当前项目运行链路（启动 + 一次推理）不出现来自 `MOSS-TTS-Nano-main` 或 `Kokoro-FastAPI-master` 的模块导入。
- 对同一输入文本与参考音频，batch 与 stream 模式均能产出非空音频字节。
- 删除 `MOSS-TTS-Nano-main` 与 `Kokoro-FastAPI-master` 后，服务仍能正常启动并完成最小推理。
