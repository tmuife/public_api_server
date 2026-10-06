## Context

`media_tool` 当前是 uv 管理的 Python 3.12 空项目，仅有打印欢迎信息的入口。用户需要把 `swap_face_server` 与 `enhance_face_server` 的核心能力抽取为独立的图片、视频命令行工具，最终在 macOS M1 上运行。用户已提供目标环境：macOS `26.6.2 (25G83)`、`32 GB` 内存。

已确认的代码和资产：

- `swap_face_server/app/services/swap_face_service.py` 支持 InsightFace 与 YuNet，使用归一化 embedding 的欧氏距离，当前阈值为 `1.2`；命中第一张脸后立即返回。两个检测器在旧初始化代码中都会被创建。
- 旧服务的 `source_face` 表示新身份、`target_face` 表示被查找身份，和用户本次定义相反。新配置采用用户定义，内部使用 `reference_faces`、`replacement_face`、`detected_faces`。
- `enhance_face_server/app/services/swap_face_service.py` 使用 GFPGAN，默认整帧再次检测并增强所有脸。
- 本地已发现 YuNet、`w600k_r50.onnx`、`inswapper_128.onnx`、`GFPGANv1.4.pth`，以及旧 GFPGAN 的额外检测、解析权重；未在两个项目模型目录发现 SCRFD `det_10g.onnx`。
- 旧项目锁定 Python 3.10、NumPy 1.23.5 和旧 PyTorch；`onnxruntime-silicon==1.16.3` 缺少 Python 3.12 arm64 安装包。BasicSR 1.4.2 引用了新版 torchvision 已移除的接口。

本变更文档正文使用中文，保留 OpenSpec 模板标题、Requirement/Scenario、MUST 和 WHEN/THEN 等格式标记，以供工具解析。用户已确认仅增强成功换脸的人脸、未匹配素材原样复制，并同意先使用设计中的初始参数；硬件信息和仍需实施验证的事项集中记录在文末。参考身份范围沿用“任意参考脸匹配”的现有设计。

## Goals / Non-Goals

**目标：**

- 将 `media_tool` 单独复制到另一台机器后，在准备依赖与本地模型的情况下可以运行；没有兄弟项目路径依赖或服务依赖。
- 同时支持 InsightFace/SCRFD 和 YuNet，并通过 `.env` 的 `detect_method` 选择。
- 对图片和每个视频帧，匹配多个参考脸，替换所有匹配脸，再增强成功替换的人脸。
- 保留输入目录层次，保护原始素材，提供可检查的输出、错误报告和安全的任务清理。
- 保留视频的展示时间和音画偏移，兼容 SDR 可变帧率素材，并控制 M1 上的内存占用。
- 建立 CPU 正确性基线，并在目标 M1 上独立评估 CoreML 和 MPS。

**本次不包含：**

- Web 服务、UI、数据库、远程任务队列、多人参考身份分别映射多个替换身份。
- 动态切换检测器、逐帧热更新配置、完整人脸跟踪、遮挡重建和帧级断点续跑。
- 动画图片、多页图片处理及 HDR 色调映射；这些输入需要明确报告，不隐式截取或当作普通 SDR 解码。
- 默认加载背景超分或旧 GFPGAN 的整帧检测网络。

## Decisions

### 1. 抽取最小推理实现，避免旧服务与旧训练依赖

选择本地、显式的模型推理模块：保留 SCRFD 后处理、ArcFace 对齐和预处理、INSwapper 身份映射与融合，以及 GFPGAN clean 推理网络。识别、检测和换脸使用官方 `onnxruntime`，GFPGAN 使用 PyTorch。必须保持模型输入语义与权重兼容，并保留所抽取上游代码的来源和许可证。

GFPGAN 网络中训练注册器等依赖可以移除，必要的初始化和图像转换代码收进推理模块。不修改 `site-packages`，不通过全局 monkey patch 修复旧包，不引入完整 BasicSR 数据增强/训练模块。INSwapper 必须保留 `emap` 身份映射及归一化，不能直接把 ArcFace embedding 当作模型 latent。

备选方案是直接安装 InsightFace 0.7.3 与 GFPGAN 1.3.8。该方案包装代码少，但会带来 InsightFace 源码构建和 BasicSR/torchvision 兼容负担，因此本次采用有边界的推理代码抽取。必须用相同图片、相同权重与旧算法做对照验证，避免抽取改变对齐、通道顺序或数值预处理。

依赖选择以 Python 3.12 与 macOS arm64 的安装包和真实试推理为依据，生成项目自己的 `uv.lock`。限制首版 Python 范围为 `>=3.12,<3.13`。不直接沿用旧项目的版本组合，也不将未实测的最新版本视为已兼容。

### 2. 两个检测器统一输出，共享识别模型

配置约定：

```dotenv
detect_method=insightface
# 切换为 yunet 使用 OpenCV YuNet
```

仅接受 `insightface` 与 `yunet`，默认 `insightface`，保持旧服务默认检测路线。旧值 `insight`、`yu` 不引入别名，错误配置明确提示有效值。

| 检测器 | 实现 | 必需检测权重 | 后端 |
|---|---|---|---|
| `insightface` | 抽取 InsightFace 的 SCRFD 预处理、输出解码和 NMS | `detection/insightface/det_10g.onnx` | ONNX CPU/CoreML |
| `yunet` | OpenCV `FaceDetectorYN` | `detection/yunet/face_detection_yunet_2023mar.onnx` | OpenCV CPU |

两者输出同一 `DetectedFace` 数据结构：原始图坐标中的浮点 `bbox=[x1,y1,x2,y2]`、形状为 `(5,2)` 的浮点关键点、检测置信度。关键点顺序固定为图像中的左眼、右眼、鼻尖、左嘴角、右嘴角，并校验各模型的输出转换。无检测结果统一返回空列表；不把推理异常转换成空列表。

两者都使用 `recognition/w600k_r50.onnx` 提取特征。参考脸、替换脸和输入素材使用本次运行选定的同一检测器，识别模型只加载一份。检测器选择通过一个小工厂函数完成，不引入插件注册框架。

`detection_max_side=640`：SCRFD 使用对应尺寸的等比缩放、填充与坐标逆变换；YuNet 限制检测图最长边并正确映射回原图。原始图片分辨率保留给换脸和输出。`detection_score_threshold=0.6` 控制最低检测置信度，NMS 初始值固定为 `0.3`。

只创建选定的检测器和会用到的模型。缺少所选检测器权重时预检查失败，不偷偷切换检测方法；缺少未选择的检测器权重时仍可运行。切换检测方法需要重新提取参考特征，不跨检测器复用缓存。

备选方案是直接加载完整 `buffalo_l` 模型包，会加载不必要的年龄、性别等模型并引入默认下载路径，因此不采用。检测器间关键点差异会影响特征距离，不能保证阈值效果相同，校准与报告必须注明所选检测器。

### 3. 配置集中加载并提前校验

默认读取当前工作目录的 `.env`，可通过 `--env` 指定。所有相对路径相对于该文件的父目录解析。优先级为默认值 < `.env` < 系统环境变量 < 显式命令行参数。配置在启动时生成一次，不在每帧读取环境变量。

首版配置示例：

```dotenv
input_material_path=./materials/input
work_dir=./work
output_material_path=./materials/output
source_face=./faces/source
target_face=./faces/target
models_path=./models

thresholds=1.25
detect_method=insightface
detection_score_threshold=0.6
detection_max_side=640

onnx_provider=auto
enhance_device=auto
enhance_enabled=true
enhance_blend=0.7

video_encoder=libx264
video_crf=18
video_preset=medium
image_jpeg_quality=95

unmatched_action=copy
overwrite=false
on_error=continue
cleanup_work_dir=true
log_level=INFO
```

校验枚举和数值范围：`0 < thresholds <= 2`、`0 < detection_score_threshold <= 1`、正整数检测尺寸、`0 <= enhance_blend <= 1`，以及对应编码器的合法参数。`unmatched_action` 接受 `copy|skip`，`on_error` 接受 `continue|stop`。

路径使用解析后的实际路径检查。输入、输出、工作目录两两不能相等或互为祖先；输入扫描也不能包含参考脸、替换脸或模型目录。输出和工作目录按需创建。真实 `.env`、运行数据和模型权重不进入 Git。

模型布局：

```text
models/
├── detection/
│   ├── insightface/det_10g.onnx
│   └── yunet/face_detection_yunet_2023mar.onnx
├── recognition/w600k_r50.onnx
├── swap/inswapper_128.onnx
└── enhance/GFPGANv1.4.pth
```

只检查所选检测器、识别、换脸，以及开启增强时的增强权重。日常运行不隐式下载模型，也不访问 `~/.insightface` 或旧项目权重。

### 4. 参考注册与身份匹配采用明确语义

`source_face` 递归读取静态参考图片，每张有效图片必须只有一张脸；无脸、多脸、损坏图片或不支持的图片类型均报告文件并使预检查失败。目录可以包含同一人的多个角度，也可以包含多个待替换身份，所有有效 embedding 构成一个参考集合。

`target_face` 递归扫描后必须只有一张有效静态图片，且只有一张脸；不隐式选择第一张或最大脸。非图片文件忽略，但不支持或损坏的候选图片不能被当作有效替换图。

参考、替换和素材特征都进行 L2 归一化，零向量或非有限值视为特征提取失败。对于素材中的每张脸，计算与全部参考特征的最小欧氏距离，严格小于 `thresholds` 才匹配。每张素材脸最多替换一次，记录最近参考与最小距离。

选择最小距离而非均值特征，以同时支持同人多角度与多个人。用户同意先使用阈值 `1.25`、检测尺寸 `640` 和增强融合比例 `0.7`。初始 `1.25` 较旧配置 `1.2` 宽松，但不是匹配概率，也不是经过验证的最终阈值；需要两种检测方法分别在实际正、负样本上标定。

### 5. 一帧完成全部匹配，再换脸、增强

```text
原始帧 → 所选检测器 → ArcFace 特征 → 全部匹配结果
       → 替换全部匹配脸 → 增强成功换脸的局部区域 → 最终帧
```

全部身份判断基于原始帧，避免换脸后的身份影响后续匹配。去除旧实现的首个命中即返回行为。

换脸使用 INSwapper 128 对齐模板、身份映射与贴回算法。增强使用 GFPGAN 自己的五点对齐模板，将换脸后的对应区域对齐到 `512×512`，推理后逆变换并以局部羽化遮罩贴回。不能把 ArcFace/INSwapper 的裁剪仅放大后当作 GFPGAN 对齐结果。

`enhance_blend` 是增强图与换脸后图的像素融合比例，不直接传作 GFPGAN 的 `weight`。按用户确认的范围，首版只增强成功换脸的区域。默认开启增强；关闭时允许仅换脸。

采用 FP32、单脸批次、推理模式和固定噪声，避免每帧随机噪声变化。没有匹配脸时跳过换脸与增强；任何推理失败使当前文件失败，不静默返回原帧，也不把仅换脸结果报告成已增强成功。

### 6. 静态图片与视频共用帧处理，媒体读写分开

Pillow 处理静态 JPEG、PNG、WebP、BMP 和单页 TIFF，可通过 `pillow-heif` 支持静态 HEIC。解码应用 EXIF 方向；模型侧统一 BGR，输出方向元数据不能导致重复旋转。透明通道单独保留。按颜色信息转换到明确的 SDR/sRGB 处理空间，不原样附带不再匹配的方向或色彩配置。

处理后的常见图片保持原文件名和格式；HEIC 输出 JPEG，使用完整原名加 `.jpg`，例如 `a.heic.jpg`。动画图片和多页 TIFF 明确作为不支持素材报告。未匹配图片按 `unmatched_action` 原样复制或跳过，复制不触发重新编码。

PyAV 顺序解码视频并编码临时无声视频；FFprobe 获取主视频流、音轨、旋转、颜色信息和起始时间；FFmpeg 从原输入读取音轨并与临时视频合并，避免音轨必须先落成独立文件。

视频使用原始帧 `PTS × time_base` 作为展示时间，以共同原点换算到编码器时间基，正确处理编码包和封装时间基变化。音轨使用同一原点并保留相对偏移。不能用平均 FPS 重建可变帧率时间线，不能独立把音频、视频均强制归零，也不能用 `-shortest` 掩盖同步问题。缺失或不可用的必要时间戳先明确报错，不隐式采用 CFR。

第一条主视频流作为首版处理对象，保留所有音轨顺序。目标 MP4 支持的音轨优先复制，否则转 AAC。无音轨输入不创建伪音轨。显式处理旋转，输出正确的展示尺寸并清除已应用的旋转标记；保持必要的像素宽高比。奇数尺寸若需要 `yuv420p` 对齐，显式补边并记录，不静默裁剪。字幕和附加视频流不纳入输出。

默认 H.264 MP4、`libx264`、CRF 18、preset medium。匹配视频：原为 MP4 时保留文件名，其它容器追加 `.mp4`，如 `a.mov.mp4`。未匹配视频必须检查完整视频后才能判断，丢弃临时编码结果并按配置原样复制或跳过；复制保留原名和容器。

只保留当前帧及有界编码缓冲，不将全部帧或原始视频字节读入内存，不生成整段图片序列。首版拒绝 HDR 素材并记录原因，后续色调映射另行设计。

### 7. 文件级调度与原子发布

递归扫描候选文件，顺序稳定，不跟随符号链接，非素材文件忽略。扩展名用于筛选，实际解码确认文件类型。保持输入相对目录和目录层次，不修改原文件。

扫描阶段同时建立处理输出和未匹配复制输出的候选路径，检查不同输入之间的路径冲突，并考虑 macOS 常见大小写不敏感文件系统。例如 `a.mov` 的处理输出可能与输入 `a.mov.mp4` 的复制输出冲突；这种情况在处理前报告并终止，不覆盖或依赖处理顺序解决。默认已有输出跳过并单独记录，`overwrite=true` 才替换已有结果。

每次运行建立 `work_dir/<run_id>/`，每个文件建立独立子目录。编码结果关闭、校验后，复制到最终输出同目录的临时文件，再用原子重命名发布，以处理工作目录与输出目录跨文件系统的情况。失败不发布半成品，原有输出保持有效。

`on_error=continue` 对文件级错误继续处理其它文件，但最终返回非零状态；`stop` 在首个文件错误后停止。配置、模型初始化等全局错误始终在批处理前退出。正常退出与中断都关闭模型之外的媒体资源和子进程，并清理本次创建的目录。不能递归清空用户配置的整个 `work_dir`。

运行报告存放于 `output_material_path/.media-tool-reports/<run_id>.jsonl`，不放在会清理的临时目录；记录输入、输出、文件状态、检测器、实际后端、帧与匹配/替换数量、耗时、错误和本次处理参数。状态区分成功、未匹配复制、跳过和失败。报告不保存原始人脸图片或 embedding。

本次不实现独立 `resume` 命令。重新执行可以根据已有输出跳过，但必须标明这是存在性跳过，不保证旧输出由同样配置生成，也不宣称完成可验证的断点续跑。

### 8. M1 后端选择与诊断

`onnx_provider=cpu|coreml|auto`，作用于 SCRFD、识别和换脸 ONNX 模型，YuNet 继续使用 CPU。`enhance_device=cpu|mps|auto`，只作用于 GFPGAN。

CPU 模式是正确性基线。`auto` 检查可用提供器并分别试运行每个模型；CoreML/MPS 的已识别兼容问题可切换 CPU，记录模型名称、原因和最终后端，不能把所有异常一概当作设备问题。显式指定 `coreml` 或 `mps` 时不可用则明确失败；允许的 ONNX CPU 算子分区必须说明，注册 CoreML 不等于全模型被加速。

CoreML 的效果由模型分别评估，不假定 SCRFD、ArcFace 与 INSwapper 都更快。MPS 可使用经过验证的算子回退，但涉及环境变量时必须在导入 PyTorch 前加载配置；不依赖它解决所有数值类型或内存问题。

单进程、单素材执行，模型与参考特征加载一次，提前计算替换身份 latent。默认单脸增强，不默认 FP16 或多进程；在目标 Mac 上记录启动时间、稳态帧耗时、端到端 FPS 和峰值内存后，再调整线程数与编码选项。VideoToolbox 是可评估的编码选项，不将 `libx264` 的 CRF 参数直接套用到硬件编码器。

### 9. 命令行与模块组织

```bash
uv sync --frozen
uv run media-tool check
uv run media-tool run --dry-run
uv run media-tool run
uv run media-tool run --env /path/to/.env
```

`check` 校验配置、目录、所选模型、参考图、FFmpeg/FFprobe 及真实试推理，并报告实际后端。`run --dry-run` 检查输入清单、可探测的格式和输出映射，不执行素材换脸，不预测匹配数量，不创建最终媒体输出。普通 `run` 必须执行相同的必要预检查。

使用 `src/media_tool/`，按 CLI、配置、runner、pipeline、检测/识别、匹配、换脸、增强、对齐、图片 IO、视频 IO 划分模块。两个检测实现放入 `detectors/insightface.py` 与 `detectors/yunet.py`，只抽取共同的结果类型和选择函数；第三方必要网络放入 `vendor/` 并保留许可。保持显式调用，不增加通用任务框架。

## Risks / Trade-offs

- [SCRFD 权重尚未在本地发现] → 在准备任务中明确取得并放入本地模型目录；默认 InsightFace 模式缺失时预检查失败，YuNet 模式只要求自己的权重。
- [抽取代码可能改变模型语义] → 对照原始预处理、特征映射、关键点模板和贴回结果；保留上游版本及许可证，增加真实权重烟雾测试。
- [阈值放宽与多参考图会增加误匹配] → 初始值标记为待校准，分别记录两种检测器正、负样本距离，使用实际素材决定最终值。
- [逐帧识别可能闪烁，局部羽化可能不及解析遮罩] → 固定增强噪声、验收连续视频；必要时单独增加跟踪或本地解析权重，不承诺首版完整时序稳定。
- [CoreML 分图或 MPS 算子限制] → 每模型诊断、CPU 基线、明确回退原因和 M1 上的真实性能比较。
- [视频时间基、起始偏移和旋转处理错误] → 合成 CFR/VFR、非零音轨偏移、无音轨及旋转样本，使用 FFprobe 断言时间线和可解码性。
- [目标机器有 32 GB 内存，但模型、帧缓冲和临时视频仍可能造成资源压力] → 保持单模型实例、单素材、单脸增强和有界缓冲；长视频运行前估算工作磁盘需求，记录峰值内存，不因内存容量增加而默认扩大并发。
- [目标 macOS 26.6.2 上的依赖与加速兼容尚未实测] → 提交锁文件前明确最低支持版本，并在用户提供的 `26.6.2 (25G83)`、32 GB M1 环境安装与试推理；Linux 只承担通用逻辑验证。

## Migration Plan

1. 建立项目包和配置、CLI 入口，选定可锁定的 Python 3.12 arm64 依赖。
2. 准备项目独立模型目录，复制已有权重并补齐 SCRFD 权重；记录来源及校验信息，不提交大权重文件。
3. 实现两种检测器、共同特征提取、参考注册与匹配，再接换脸与局部增强。
4. 完成静态图片、批处理输出，再接视频时间线与音轨；全部验证完成后更新中文 README。
5. 在 macOS `26.6.2 (25G83)`、32 GB M1 环境做安装、真实素材与性能验收，分别校准两种检测器阈值。

这是新增工具，无数据库迁移和旧服务切换。回退可停止使用新工具并保留原始素材；默认无覆盖，最终文件按单文件事务发布，旧服务继续原有行为。

## Open Questions

用户回复已记录，当前没有需要补充回答才能推进实现的问题。

**已确认：**

- 目标环境：M1 平台，macOS `26.6.2 (25G83)`，`32 GB` 内存。
- 处理策略：仅增强成功换脸的人脸；未匹配素材原样复制。参考身份范围沿用现有“任意参考脸匹配”的设计。
- 初始参数：先使用 `thresholds=1.25`、`detection_max_side=640`、`enhance_blend=0.7`，保留通过 `.env` 调整的能力。

**仍需实施验证：**

- 明确锁定依赖的最低 macOS 支持版本，并在上述目标 M1 环境完成安装、模型试推理和性能验收；用户提供硬件信息不等于兼容性已验收。
- 两种检测器分别用真实素材评估阈值、检测尺寸和增强融合比例。当前参数是用户同意先使用的初始值，不是已经验证的质量结论。
