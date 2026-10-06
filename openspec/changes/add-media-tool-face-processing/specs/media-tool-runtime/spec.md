## ADDED Requirements

### Requirement: 项目可以独立运行

系统 MUST 在 Python 3.12 与 uv 管理的独立 `media_tool` 项目内提供完整运行入口、锁定依赖和必要推理代码；运行时不导入兄弟项目、不调用旧服务、不访问旧项目模型目录。抽取的上游代码 MUST 保留来源与许可证。

#### Scenario: 脱离原仓库运行
- **WHEN** 将 `media_tool` 单独复制到新目录并准备其依赖、模型、配置和输入素材
- **THEN** `uv run media-tool check` 与 `uv run media-tool run` 可以在不存在两个旧项目的情况下执行

### Requirement: 环境配置统一解析

系统 MUST 从默认当前目录的 `.env` 或 `--env` 指定文件读取配置，并按默认值、配置文件、系统环境变量、显式命令行参数的顺序覆盖。路径 MUST 相对于配置文件父目录解析。配置 MUST 在本次运行开始时固定，不在帧处理中重新读取。

#### Scenario: 指定其它目录的配置
- **WHEN** 从目录 A 执行命令，并通过 `--env` 指定目录 B 中的配置，配置包含相对输入路径
- **THEN** 输入路径相对于目录 B 解析，而不是目录 A

#### Scenario: 环境变量覆盖配置文件
- **WHEN** `.env` 设置 `detect_method=insightface`，系统环境变量设置 `detect_method=yunet`
- **THEN** 本次运行选择 YuNet，并在诊断中报告该选择

### Requirement: 配置值与目录安全校验

系统 MUST 校验所有必需路径、支持的枚举和参数范围，包括 `0 < thresholds <= 2`、合法检测置信度、正检测尺寸和 `0 <= enhance_blend <= 1`。系统 MUST 在扫描前拒绝输入、输出、工作目录两两相等或互相包含的配置，以及参考脸、替换脸或模型目录被包含在输入目录中的配置；判断 MUST 基于解析后的实际路径。

#### Scenario: 输出目录嵌入输入目录
- **WHEN** `output_material_path` 是 `input_material_path` 的子目录
- **THEN** 系统在处理任何素材前以非零状态退出，并说明目录包含关系

#### Scenario: 阈值不合法
- **WHEN** `thresholds` 不是有限数值或不在允许范围内
- **THEN** 系统明确报告配置错误，不启动素材处理

### Requirement: 按需加载本地模型

系统 MUST 仅从 `models_path` 加载所选检测器、ArcFace 识别、INSwapper 换脸及启用时的 GFPGAN 增强模型，并在运行前验证存在性、输入输出规格和可用性。系统 MUST 不隐式下载模型或依赖用户级 InsightFace 缓存。关闭增强时 MUST 不要求 GFPGAN 权重存在。

#### Scenario: 仅准备 YuNet 模式模型
- **WHEN** 选择 YuNet，准备了 YuNet、识别、换脸及所需增强权重，但没有 SCRFD 权重
- **THEN** 预检查不因缺少 SCRFD 而失败，也不创建 SCRFD 会话

#### Scenario: 所选模型缺失
- **WHEN** 选择 InsightFace，但 `detection/insightface/det_10g.onnx` 不存在
- **THEN** 系统列出缺失路径并退出，不自动下载或切换到 YuNet

### Requirement: 推理后端选择可观察

系统 MUST 支持 `onnx_provider=cpu|coreml|auto` 与 `enhance_device=cpu|mps|auto`。`auto` MUST 逐模型检查可用性并试推理，识别出的后端兼容失败可以回退 CPU，同时报告原因与实际后端；未识别的模型或数据错误 MUST 作为错误报告。显式选择不可用的加速后端 MUST 失败。YuNet MUST 使用 CPU，并区分 ONNX 后端与增强后端。

#### Scenario: 自动模式无加速后端
- **WHEN** `auto` 模式运行于没有 CoreML、MPS 的环境
- **THEN** 对应模型使用 CPU，并明确报告实际后端

#### Scenario: 显式选择不可用 MPS
- **WHEN** 配置 `enhance_device=mps`，但当前 PyTorch 或系统不支持 MPS
- **THEN** 预检查失败，不静默改用 CPU

#### Scenario: CoreML 只执行部分模型算子
- **WHEN** ONNX 会话使用 CoreML 与 CPU 算子分区
- **THEN** 诊断不将“注册 CoreML”宣称为全模型硬件加速，并报告提供器配置

### Requirement: 命令行检查与执行

系统 MUST 提供 `check`、`run --dry-run`、`run` 和 `--env` 配置选择。`check` MUST 检查配置、所需模型、参考图、FFmpeg/FFprobe 和试推理；`run --dry-run` MUST 检查素材清单和输出映射，不执行换脸、不产生最终媒体、不预测匹配数量；`run` MUST 在处理前执行必要预检查。失败 MUST 返回非零退出状态。

#### Scenario: 执行预览
- **WHEN** 对有效配置执行 `uv run media-tool run --dry-run`
- **THEN** 输出计划处理的素材和目标路径，且不生成转换后的图片或视频

#### Scenario: 编解码工具缺失
- **WHEN** 执行检查或批处理，但无法找到 FFmpeg 或 FFprobe
- **THEN** 系统报告缺失工具并在处理前退出

### Requirement: macOS arm64 安装与验收

项目 MUST 提供针对 Python 3.12、macOS arm64 可安装的锁定依赖与中文安装说明，并声明实际支持的最低 macOS 版本。目标验收环境为用户提供的 M1 平台、macOS `26.6.2 (25G83)`、32 GB 内存。M1 兼容与性能结论 MUST 基于该目标环境的真实安装、模型试推理和素材处理，不能以 Linux 或安装包元数据检查代替。

#### Scenario: M1 环境安装
- **WHEN** 在目标 macOS `26.6.2 (25G83)`、32 GB M1 环境准备本机工具和模型后执行 `uv sync --frozen`
- **THEN** 可以运行两种检测器的检查和真实素材处理，并记录后端与性能结果
