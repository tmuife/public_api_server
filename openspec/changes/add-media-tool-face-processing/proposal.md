## Why

现有 `swap_face_server` 与 `enhance_face_server` 已具备人脸检测、身份匹配、换脸和增强的核心能力，但其服务接口、依赖版本和媒体处理方式不适合直接用于 macOS M1 上的离线批处理。需要在现有 uv 管理的 Python 3.12 项目 `media_tool` 中建立独立运行的图片、视频处理工具，并允许通过配置选择 InsightFace 或 YuNet 检测器。

## What Changes

- 从两个旧项目及其所用上游推理实现中抽取必要的模型推理、对齐和融合代码到 `media_tool`，保留来源与许可证；运行时不导入旧项目、不调用旧服务、不访问旧项目模型路径。
- 通过 `.env` 管理输入、输出、工作、参考脸、替换脸和模型目录，以及匹配阈值、检测方法、推理设备和输出参数。
- 首版同时支持 InsightFace 的 SCRFD 检测器与 OpenCV YuNet 检测器，使用 `detect_method=insightface|yunet` 选择；默认 `insightface`，两者使用相同的本地 ArcFace 识别模型。
- 递归读取图片和视频，按输入相对目录生成输出；匹配任意参考脸的人脸均替换成指定身份，并在换脸后增强成功替换的人脸。
- 保留现有归一化特征的欧氏距离定义，采用用户同意先使用的 `thresholds=1.25` 作为待标定的初始值，支持单帧多个人脸匹配与替换。
- 视频采用逐帧处理，保留可变帧率时间戳、音画相对起始偏移与原音轨；控制临时文件和内存占用。
- 提供 `check`、`run --dry-run` 和 `run` 命令，以及文件级错误报告、原子输出和安全清理。
- 针对 Python 3.12、macOS arm64 验证依赖，并提供 ONNX CPU/CoreML、PyTorch CPU/MPS 后端选择与可观察的回退。

## Capabilities

### New Capabilities

- `media-tool-runtime`：独立项目、配置校验、本地模型、命令行和 M1 推理后端管理。
- `media-tool-face-detection`：可配置的 InsightFace/SCRFD 与 YuNet 检测，统一人脸框、五点关键点和置信度输出。
- `media-tool-face-transformation`：参考脸注册、归一化特征匹配、多脸换脸以及换脸后局部增强。
- `media-tool-batch-processing`：素材递归扫描、静态图片读写、输出映射、文件级容错、报告和临时目录管理。
- `media-tool-video-processing`：流式视频处理、时间戳保留、旋转处理、音轨合并和视频输出校验。

### Modified Capabilities

无。当前没有既有 OpenSpec 主规格，本变更不修改两个旧服务的接口或行为。

## Impact

- 实现范围为 `media_tool/`；两个旧项目仅作为代码与模型来源，不修改其实现，也不涉及其它子项目。
- 新增项目自己的 Python 依赖、`uv.lock`、命令行入口、`.env.example`、必要测试和中文使用说明；不提交真实 `.env` 配置。
- 核心模型为 SCRFD `det_10g.onnx`、YuNet、`w600k_r50.onnx`、`inswapper_128.onnx` 和 `GFPGANv1.4.pth`。当前已发现后四种本地模型；SCRFD 权重需要补齐，使用 YuNet 时不要求 SCRFD 权重存在。
- 视频处理需要 PyAV 以及本机 FFmpeg/FFprobe。首版处理 SDR 视频；HDR、动画图片、多页图片和完整时序跟踪不纳入本次实现。
- 用户已确认仅增强换脸成功的人脸、未匹配素材原样复制，并同意先使用阈值 `1.25`、检测尺寸 `640` 和增强融合比例 `0.7`。参考图仍沿用可覆盖一个或多个人的设计，每张参考图只有一张明确人脸，替换目录只有一张有效单脸图片；这些行为采用明确配置或校验规则，不依赖隐式选择。
- 用户提供的目标环境为 M1 平台、macOS `26.6.2 (25G83)`、`32 GB` 内存。依赖最低支持系统版本仍须明确，安装、模型兼容与实际加速效果必须在该 arm64 环境验收，不能以 Linux 检查代替。
