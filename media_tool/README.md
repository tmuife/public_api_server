# media_tool

使用本地模型递归处理图片和视频：检测人脸，与 `source_face` 参考集合匹配，将匹配脸换成 `target_face` 的身份，再增强成功换脸的人脸。

项目使用 Python 3.12 和 uv，可单独复制运行。运行时不导入 `swap_face_server`、`enhance_face_server`，不访问它们的模型目录，也不调用服务 API。首版支持 InsightFace/SCRFD 和 YuNet 两种检测方法。

## 安装与配置

锁定依赖的 Apple Silicon 安装包最低要求 **macOS 14**（由 pillow-heif 等轮子的系统要求决定）。目标机器为 M1 平台、macOS 26.6.2（25G83）、32 GB 内存；该目标机器的真实验收仍待执行。

```bash
cd media_tool
brew install uv ffmpeg
uv sync --frozen
cp .env.example .env
mkdir -p materials/input materials/output faces/source faces/target work
```

将待处理素材放入 `materials/input`，参考照片放入 `faces/source`，唯一替换照片放入 `faces/target`。每张参考图与替换图都必须恰好有一张明确人脸。参考集合可以包含一个人的多个角度或多个人；所有命中都换成同一个替换身份。

编辑 `.env`，路径相对于该文件所在目录解析。优先级为默认值 < `.env` < 系统环境变量 < 显式命令行参数。输入、输出、工作目录不得相同或互相包含；参考、替换和模型目录须与素材扫描、输出和工作目录隔离。

## 本地模型

```text
models/
├── detection/
│   ├── insightface/det_10g.onnx
│   └── yunet/face_detection_yunet_2023mar.onnx
├── recognition/w600k_r50.onnx
├── swap/inswapper_128.onnx
└── enhance/GFPGANv1.4.pth
```

本工作区已复制四个原有权重并补齐 SCRFD，模型总大小约 1.02 GiB。权重不进入 Git；通过 Git 获取项目时须另行准备上述文件，完整复制当前项目时应包含 `models/`。

`models-manifest.json` 记录当前权重的大小、SHA-256 和 SCRFD 来源，[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 记录推理代码来源及许可。检查命令验证模型规格并试推理，日常运行不会下载模型。只要求所选检测器的权重；关闭增强后不要求 GFPGAN 权重。

## 运行

```bash
# 环境、模型、参考图及真实试推理检查
uv run media-tool check

# 素材清单、类型和输出映射预览，不预测匹配结果
uv run media-tool run --dry-run

# 完整批处理
uv run media-tool run

# 指定其它位置的配置文件
uv run media-tool run --env /path/to/.env

# 临时选择 YuNet，并使用 CPU 基线
uv run media-tool check --detect-method yunet --onnx-provider cpu --enhance-device cpu
```

还可以使用 `uv run python -m media_tool` 或 `uv run python main.py`，参数与上面一致。命令失败返回非零退出状态；`Ctrl+C` 返回 130，不发布当前半成品。

## 使用 FFmpeg 截取视频素材

以下命令在 `media_tool` 目录执行，将输入文件名替换为实际路径。示例从 `00:33:55` 开始截取 4 秒，输出到 `materials/input`；`-ss` 指定起点，`-t` 指定持续时间。`-map 0:v:0` 选择第一路视频，`-map '0:a?'` 保留全部音轨，无音轨时也能运行。示例不保留字幕，`-map_chapters -1` 避免把原影片的章节信息带入短片。

### 快速截取，并将音频转成立体声

准备短测试素材时，可以复制视频流，仅重新编码音频：

```bash
ffmpeg -ss 00:33:55 -i "原视频.mkv" -t 4 \
  -map 0:v:0 -map '0:a?' \
  -c:v copy -c:a aac -ac 2 -b:a 192k \
  -map_chapters -1 \
  materials/input/input_v_stereo.mp4
```

`-c:v copy` 不重新压缩画面，速度快，但截取起点受视频关键帧限制，不能保证精确到指定帧，输出时长也可能略有差异。原视频编码须能封装到 MP4，并由播放器支持；需要统一编码时使用下面的精确截取命令。

`-c:a aac -ac 2` 将每路音轨编码为双声道 AAC，正常把 5.1 的中央等声道混入左右声道。对白通常集中在中央声道；如果播放器只播放原音轨的左右声道而未正确混音，可能听起来几乎没有声音。

如果要原样保留音视频压缩数据及原声道布局，可将上述编码选项替换为 `-c copy`。它不会主动删除声音，也不会把 5.1 转成立体声；音视频编码都须与 MP4 兼容。

### 精确截取，并统一为 H.264 与立体声 AAC

需要更准确的起点或更广泛的播放兼容性时，重新编码视频和音频：

```bash
ffmpeg -ss 00:33:55 -i "原视频.mkv" -t 4 \
  -map 0:v:0 -map '0:a?' \
  -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p \
  -c:a aac -ac 2 -b:a 192k \
  -map_chapters -1 \
  materials/input/input_v_precise.mp4
```

重新编码时，FFmpeg 默认启用精确寻址，会解码并丢弃起点前的内容，截取精度受原视频帧时间限制。此方式比流复制慢，并会重新压缩画面；示例适用于 SDR 素材，不包含 HDR 到 SDR 的色调映射。

### 已截取片段的音频检查与立体声转换

先检查片段是否有音轨及其声道布局：

```bash
ffprobe -v error -select_streams a \
  -show_entries stream=index,codec_name,sample_rate,channels,channel_layout,duration \
  -of json materials/input/input_v.mp4
```

`streams` 为空表示没有音轨；`channels=6`、`channel_layout=5.1` 表示六声道，不能据此判断音频是否静音。如果播放器听不到声音，可先尝试其他播放器，或将已有片段转成立体声，无需重新截取或换脸：

```bash
ffmpeg -i materials/input/input_v.mp4 \
  -map 0:v:0 -map '0:a?' \
  -c:v copy -c:a aac -ac 2 -b:a 192k \
  -map_chapters -1 \
  materials/input/input_v_stereo.mp4
```

转换不会给无音轨的片段生成声音。完成后将要处理的片段留在输入目录即可；工具会递归处理目录中的所有支持素材，测试时可将其它片段移到输入目录外。

## 常用参数

完整示例见 `.env.example`。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `detect_method` | `insightface` | `insightface` 使用 SCRFD；`yunet` 使用 OpenCV CPU；只加载选中的检测器 |
| `thresholds` | `1.25` | 与参考集合的最小归一化欧氏距离严格小于此值才匹配；越大越宽松 |
| `detection_max_side` | `640` | 检测输入尺寸，至少 32 且为 32 的倍数；原图用于换脸、融合和输出 |
| `detection_score_threshold` | `0.6` | 最低检测置信度 |
| `onnx_provider` | `auto` | `cpu`、`coreml`、`auto`，作用于 SCRFD、识别和换脸 |
| `enhance_device` | `auto` | `cpu`、`mps`、`auto`，只作用于 GFPGAN |
| `enhance_enabled` | `true` | 仅增强成功换脸的人脸；关闭时只换脸 |
| `enhance_blend` | `0.7` | 增强结果与换脸后区域的像素融合比例，0..1 |
| `video_crf` / `video_preset` | `18` / `medium` | H.264 `libx264` 编码质量与速度 |
| `image_jpeg_quality` | `95` | JPEG/WebP 输出质量 |
| `unmatched_action` | `copy` | 未匹配素材按字节原样复制；也可显式选择 `skip` |
| `overwrite` | `false` | 已有候选输出默认跳过；不验证其是否由同样配置生成 |
| `on_error` | `continue` | 单文件失败后继续，也可选择 `stop`；有失败的批次始终返回非零状态 |
| `cleanup_work_dir` | `true` | 只清理本次工具创建的任务目录，保留用户原有文件 |

`1.25`、`640` 和 `0.7` 是已同意先使用的初始值，仍需按实际素材校准。两种检测器共用同一 ArcFace 模型，但关键点差异会影响特征距离；切换检测器会重新注册参考特征。`DEBUG` 日志可查看最近参考与距离，报告不保存 embedding。

`auto` 会逐模型检查后端并试推理；已识别的加速兼容问题会报告并切换 CPU。显式选择不可用 CoreML/MPS 会失败。ONNX 会话注册 CoreML 不代表所有算子都使用加速器；实际提供器和增强设备会写入检查结果及运行报告。

## 素材与输出规则

- 支持静态 JPEG、PNG、WebP、BMP、单页 8 位 TIFF、静态 8 位 HEIC/HEIF；处理 EXIF 方向、ICC 到 sRGB 的转换及透明通道。
- HEIC/HEIF 处理结果为完整原名加 `.jpg`；其它支持图片保留原名和格式。
- 视频首版支持可解码 SDR 素材，处理第一条主视频流，保留逐帧展示时间、所有音轨顺序及音画起始偏移；兼容音轨复制，否则转 AAC。旋转应用到像素，输出不会重复旋转。
- 处理视频输出 H.264 MP4：原 MP4 保留文件名，其它容器追加 `.mp4`，如 `a.mov.mp4`。必要时奇数尺寸补一像素边，并记录日志。
- 没有匹配的视频仍需检查完整视频；确认后丢弃临时编码结果，复制原文件，避免重新压缩。
- 输出保持相对子目录，预先检查处理/复制路径冲突及大小写冲突。不会按文件顺序解决冲突或隐式覆盖。
- 输出先校验，再在输出目录中原子发布。临时目录可位于其它文件系统；编码或合并失败不损坏已有结果。
- HDR、动画 GIF/WebP、GIF/AVIF、多页 TIFF、高位深图片、不可靠时间戳、中途变化的分辨率首版不支持，会明确报告。字幕和附加视频流不进入输出。

单进程顺序处理，模型与参考特征只加载一次，使用有界帧缓冲；不把整个视频读入内存，不拆成完整图片序列。GFPGAN 使用固定噪声；首版没有时序跟踪，遮挡、侧脸和阈值边界仍可能导致帧间换脸变化。

运行报告位于 `output_material_path/.media-tool-reports/<run_id>.jsonl`，记录状态、路径、检测器、参数、实际后端、帧/匹配/替换数、耗时、端到端 FPS 和进程峰值内存。清理临时目录不会删除报告。已有输出跳过仅表示文件存在，不等同于经过验证的断点续跑。

## 验证

```bash
uv run pytest -q
uv run ruff check src tests main.py
uv run ruff format --check src tests main.py

# 真实权重测试需显式启用，并提供两张有效的公开/自有单脸测试图
MEDIA_TOOL_MODEL_TESTS=1 \
MEDIA_TOOL_TEST_FACE=/path/to/reference.jpg \
MEDIA_TOOL_TEST_TARGET=/path/to/replacement.jpg \
uv run pytest tests/test_models.py -q
```

已在 Linux x86_64 完成双检测器、ArcFace、INSwapper、GFPGAN 的 CPU 真实推理，公开图片和合成带音轨短视频的完整 CLI 处理，以及 CFR/VFR、多音轨、音画偏移、旋转、末帧才匹配、未匹配字节复制、错误容错和输出保护测试。启用真实模型测试后共 52 项通过。对公开单脸样例的上游对照中，对齐矩阵、裁剪与 INSwapper 贴回像素一致；项目已复制到独立目录，以锁定依赖重新安装并完成双检测器检查及图片、视频处理。

M1 的 CoreML/MPS 兼容、性能、峰值内存和用户实际素材的阈值校准尚未验收。请在目标 Mac 分别运行 CPU 基线与 `auto` 模式的 `check` 和同一批短素材，保留检查输出及 JSONL 报告后比较；Linux 测试不能替代这一步。
