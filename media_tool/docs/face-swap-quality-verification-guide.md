# 侧脸换脸质量验证手册：media-tool 与 FaceFusion 独立对照

适用环境：**macOS 14 或以上、Apple Silicon、Bash、Python 3.12**。以 CPU 为基线，先验证质量，再考虑加速。Ubuntu 的安装差异见第 2 节。本手册只操作现有程序，不修改应用代码。

第一次操作的顺序：**第 2 节安装 → 第 3 节下载基础、HyperSwap 和 GFPGAN → 第 4 节准备素材 → 第 5 节定义命令 → 第 6 节逐项比较**。第 7～9 节按需执行。图片与视频取帧是二选一，不要连续执行第 4.1 和 4.2 节生成同一个文件；Mac 不执行 Ubuntu 的安装分支。

## 1. 先说明到底怎么试

**主要实验不是“media-tool 换坏了，再让 FaceFusion 重新换一次”。** 应该把同一份原始素材、同一个替换身份，分别交给两个项目处理：

```text
同一张原始侧脸图片 + 同一张替换身份照片
  ├─ media-tool → 当前流水线结果 / 关闭增强的结果
  └─ FaceFusion → 基线 / 关键点 / 不同换脸模型 / 遮罩 / 增强
```

第二次换脸会把第一次生成的错误五官当作输入，无法可靠定位问题。FaceFusion 可以只增强已有结果，但通常只能改善清晰度，不能把拉歪的鼻子、重复的眼睛或错误的侧脸朝向修好。第 9 节单独给出这种辅助试验。

这次最值得按顺序尝试的是：**关闭增强定位问题 → 关键点精修 → HyperSwap 三个变体 → 区域/遮挡遮罩 → 低强度增强 → 短视频验证**。每次只改变一个因素，选出有效项以后再组合。

两个项目的参数命名容易弄反：

| 你手里的文件 | media-tool | FaceFusion |
|---|---|---|
| 想换掉的原始人物/原图 | `input_material_path`；该人物的参考照片放在 `source_face` | `-t` / `--target-path` |
| 想换成谁：替换身份照片 | `target_face` | `-s` / `--source-paths` |
| 输出结果 | `output_material_path` 目录 | `-o`，必须包含输出文件名 |

先用**只有一张脸的原图和一张脸的替换照片**。暂时不要混入多人、长视频、HDR、HEIC、透明图片。正脸、中等侧脸、大侧脸各准备一例；先完整跑通一个最典型的失败例，再换案例。不要通过裁掉半张脸来制造侧脸。

## 2. 安装：两个项目必须使用两个环境

### 2.1 macOS Apple Silicon

先安装 [Homebrew](https://brew.sh/) 并按其提示配置 PATH。随后在终端执行：

```bash
brew install uv ffmpeg git
bash
set -e
set -o pipefail
command -v uv
command -v ffmpeg
command -v ffprobe
command -v curl
sw_vers -productVersion
uname -m
```

预期：系统版本至少 14，架构为 `arm64`。**后续全部代码块都在这一个 Bash 会话中执行**；不要直接在默认 zsh 中粘贴 Bash 数组。`set -e` 使关键命令失败时停止；若因此退出了 Bash，修正问题后按第 2.4 节恢复会话，不要继续粘贴下一步。

如果是 Ubuntu 22.04/24.04 x86_64，替代这一小节的安装命令是：

```bash
sudo apt-get update
sudo apt-get install -y git curl ffmpeg ca-certificates
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
sh /tmp/uv-install.sh
export PATH="$HOME/.local/bin:$PATH"
bash
set -e
set -o pipefail
```

macOS 不执行 `apt-get`；Ubuntu 不执行 `brew`、`sw_vers`、`open`。

### 2.2 固定路径并准备 media-tool

先把**你已经验证过的 media_tool 项目完整复制到 Mac**，包含 `uv.lock`、`models-manifest.json` 和 `models/`；不要复制旧机器的 `.venv`。若已复制了 `.venv`，在新复制的项目中将它移到其他位置，再运行安装命令。不要共享两个项目的虚拟环境。

下面只有 `MEDIA` 需要按你的实际路径修改。`LAB` 用于存放独立实验，`CASE` 是本次单图案例。示例目录如果已存在，请改成 `side02`，避免混入旧结果。

```bash
export MEDIA="$HOME/project/media_tool"    # 改成你在 Mac 上的实际绝对路径
export LAB="$HOME/face-swap-lab"
export FF="$LAB/facefusion"
export CASE="$LAB/cases/side01"
test -f "$MEDIA/pyproject.toml"
test -f "$MEDIA/uv.lock"
test ! -e "$CASE"
mkdir -p "$CASE/input" "$CASE/reference" "$CASE/replacement" \
  "$CASE/results" "$CASE/logs"

cat > "$LAB/paths.sh" <<EOF
export MEDIA="$MEDIA"
export LAB="$LAB"
export FF="$FF"
export CASE="$CASE"
EOF

cd "$MEDIA"
uv sync --frozen
"$MEDIA/.venv/bin/python" -V
```

核验复制来的权重，任何不一致都应先解决：

```bash
"$MEDIA/.venv/bin/python" - "$MEDIA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for item in json.loads((root / 'models-manifest.json').read_text())['models']:
    path = root / 'models' / item['path']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if path.stat().st_size != item['size'] or digest.hexdigest() != item['sha256']:
        raise SystemExit(f'模型不匹配：{path}')
    print('OK', item['path'])
PY
```

### 2.3 固定 FaceFusion 版本

本手册对应当前对照项目的提交 **`9cedad7d2c6f8534f20f66e425d113a655042b07`**。不要换成最新版后继续照抄参数。该版本使用 `run`，不支持旧教程中的 `headless-run`。

```bash
test ! -e "$FF"
git clone https://github.com/facefusion/facefusion.git "$FF"
git -C "$FF" checkout --detach 9cedad7d2c6f8534f20f66e425d113a655042b07
test "$(git -C "$FF" rev-parse HEAD)" = 9cedad7d2c6f8534f20f66e425d113a655042b07
uv venv --python 3.12 "$FF/.venv"
uv pip install --python "$FF/.venv/bin/python" -r "$FF/requirements.txt"
"$FF/.venv/bin/python" -c 'import cv2, numpy, onnxruntime; print("numpy", numpy.__version__); print("onnxruntime", onnxruntime.__version__); print(onnxruntime.get_available_providers())'
cd "$FF"
"$FF/.venv/bin/python" facefusion.py run --help > "$CASE/logs/facefusion-help.txt"
touch "$CASE/empty.ini"
```

输出中应有 `CPUExecutionProvider`。本版本锁定 NumPy 2.4.6、ONNX Runtime 1.30.0；media-tool 使用 NumPy 1.26.4、ONNX Runtime 1.23.2，因此不能共用 `.venv`。FaceFusion 安装以 **`requirements.txt` 为准**，不用当前工作区那个依赖为空的 `pyproject.toml`，也不用 `uv run` 自动选择环境。

### 2.4 关闭终端以后如何继续

```bash
bash
set -e
set -o pipefail
source "$HOME/face-swap-lab/paths.sh"
# 第 5 节完成后才有下面这个文件：
source "$CASE/commands.sh"
```

如果尚未完成第 5 节，只恢复 `paths.sh`，继续未完成的小节即可。重新开一个案例需要重新设置 `CASE`、准备素材和第 5 节的辅助命令文件；不用重装依赖和重复下载模型。

## 3. 模型：需要哪些、下载到哪里

### 3.1 media-tool：保持你已验证的原权重

| 模型/用途 | 本项目路径 | 约 MiB | 本手册用法 |
|---|---|---:|---|
| SCRFD 检测 | `models/detection/insightface/det_10g.onnx` | 16.1 | media-tool 基线必需 |
| ArcFace 身份识别 | `models/recognition/w600k_r50.onnx` | 166.3 | 必需 |
| INSwapper 128 换脸 | `models/swap/inswapper_128.onnx` | 528.6 | 必需 |
| GFPGAN 1.4 增强，PyTorch 权重 | `models/enhance/GFPGANv1.4.pth` | 332.5 | 开增强时必需 |
| YuNet 检测 | `models/detection/yunet/face_detection_yunet_2023mar.onnx` | 0.2 | 本手册不切换；完整复制时仍保留 |

为了比较你当前已经验证的实现，**从原环境复制这些文件并用第 2.2 节的清单校验**。不要把下面 FaceFusion 的 ONNX 改名塞进 media-tool：两个项目的识别、换脸权重导出文件并不完全相同，GFPGAN 的 `.pth` 和 `.onnx` 更不能互换。

### 3.2 FaceFusion：完整下载表

所有文件直接放在 **`$FF/.assets/models/`**，不建模型名子目录。每个 `.onnx` 同时下载同名 `.hash`。点击表中的“权重”和“校验”可手动下载；脚本下载见下一节。MiB 按 1024² 字节计算，不含运行内存。

| 组 | 文件名（不含扩展名） | 用途 | 发布标签 | 约 MiB | 下载 |
|---|---|---|---|---:|---|
| 基础 | `scrfd_2.5g` | 人脸检测 | models-3.0.0 | 3.1 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/scrfd_2.5g.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/scrfd_2.5g.hash) |
| 基础 | `2dfan4` | 关键点精修；关闭精修时也要通过预检查 | models-3.0.0 | 93.4 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/2dfan4.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/2dfan4.hash) |
| 基础 | `fan_68_5` | 五点到 68 点的估计 | models-3.0.0 | 0.9 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/fan_68_5.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/fan_68_5.hash) |
| 基础 | `arcface_w600k_r50` | 身份特征 | models-3.0.0 | 166.3 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.hash) |
| 基础 | `fairface` | 年龄/性别/族群分类的共享预检查 | models-3.0.0 | 81.2 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/fairface.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/fairface.hash) |
| 基础 | `bisenet_resnet_34` | 人脸区域解析，region 遮罩 | models-3.0.0 | 89.3 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.hash) |
| 基础 | `xseg_1` | occlusion 遮挡遮罩 | models-3.1.0 | 67.1 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_1.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_1.hash) |
| 基础 | `nsfw_1` | 共享内容分析 | models-3.3.0 | 76.7 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_1.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_1.hash) |
| 基础 | `nsfw_2` | 共享内容分析 | models-3.3.0 | 21.4 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_2.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_2.hash) |
| 基础 | `nsfw_3` | 共享内容分析 | models-3.3.0 | 341.6 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_3.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/nsfw_3.hash) |
| 基础 | `kim_vocal_2` | 共享人声模块预检查；图片实验也要求文件存在 | models-3.0.0 | 63.7 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/kim_vocal_2.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/kim_vocal_2.hash) |
| 基础 | `inswapper_128` | 换脸基线 | models-3.0.0 | 529.6 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.hash) |
| 换模型 | `hyperswap_1a_256` | 256 分辨率换脸候选 A | models-3.3.0 | 384.1 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.hash) |
| 换模型 | `hyperswap_1b_256` | 换脸候选 B | models-3.3.0 | 384.1 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1b_256.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1b_256.hash) |
| 换模型 | `hyperswap_1c_256` | 换脸候选 C | models-3.3.0 | 384.1 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1c_256.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1c_256.hash) |
| 增强 | `gfpgan_1.4` | GFPGAN 1.4，ONNX 导出 | models-3.0.0 | 324.5 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.hash) |
| 可选关键点 | `hrffa` | 2DFAN4 的对照候选 | models-3.9.0 | 34.1 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.9.0/hrffa.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.9.0/hrffa.hash) |
| 可选关键点 | `peppa_wutz` | 另一关键点候选 | models-3.0.0 | 13.1 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/peppa_wutz.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/peppa_wutz.hash) |
| 可选表情 | `live_portrait_feature_extractor` | 表情恢复：外观特征 | models-3.0.0 | 3.2 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_feature_extractor.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_feature_extractor.hash) |
| 可选表情 | `live_portrait_motion_extractor` | 表情恢复：运动/姿态 | models-3.0.0 | 107.4 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_motion_extractor.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_motion_extractor.hash) |
| 可选表情 | `live_portrait_generator` | 表情恢复：生成 | models-3.0.0 | 212.0 | [权重](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_generator.onnx) / [校验](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/live_portrait_generator.hash) |

基础组约 **1.5 GiB**；基础 + 三个 HyperSwap + GFPGAN 约 **2.9 GiB**；全表约 **3.3 GiB**。这不是内存需求；CPU 模式仍要预留模型加载、视频临时帧和环境安装空间，建议实验盘至少留 10 GiB。

`fairface`、`kim_vocal_2` 等看似与单图换脸无关，但本版本的共享预检查会要求它们。即使只用 `box` 遮罩也要备齐基础组。不要只下载换脸权重。不要使用 `force-download --download-scope lite` 来下载这张表：它仍会下载许多本实验不使用的模型。

### 3.3 下载基础组，随后按需下载可选组

先定义下载命令。默认从官方 GitHub 资源下载；如果该网络访问 GitHub 资源失败，将 `ASSET_HOST=github` 改成 `ASSET_HOST=huggingface` 再执行。两处均为 FaceFusion 官方模型仓库。

```bash
export ASSET_HOST=github
mkdir -p "$FF/.assets/models"
fetch_model() {
  local tag="$1" name="$2" base
  if [ "$ASSET_HOST" = huggingface ]; then
    base="https://huggingface.co/facefusion/$tag/resolve/main"
  else
    base="https://github.com/facefusion/facefusion-assets/releases/download/$tag"
  fi
  curl -fL --retry 3 --connect-timeout 20 --max-time 1800 \
    "$base/$name.hash" -o "$FF/.assets/models/$name.hash.part" || return 1
  curl -fL --retry 3 --connect-timeout 20 --max-time 1800 \
    "$base/$name.onnx" -o "$FF/.assets/models/$name.onnx.part" || return 1
  "$FF/.venv/bin/python" - "$FF/.assets/models" "$name" <<'PY'
import re
import sys
import zlib
from pathlib import Path
root, name = Path(sys.argv[1]), sys.argv[2]
expected = (root / f'{name}.hash.part').read_text()
crc = 0
with (root / f'{name}.onnx.part').open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        crc = zlib.crc32(block, crc)
if not re.fullmatch('[0-9a-f]{8}', expected) or f'{crc:08x}' != expected:
    raise SystemExit(f'校验失败，停止：{name}')
print('下载校验 OK', name)
PY
  if [ "$?" -ne 0 ]; then return 1; fi
  mv "$FF/.assets/models/$name.hash.part" "$FF/.assets/models/$name.hash"
  mv "$FF/.assets/models/$name.onnx.part" "$FF/.assets/models/$name.onnx"
}

for name in scrfd_2.5g 2dfan4 fan_68_5 arcface_w600k_r50 fairface \
  bisenet_resnet_34 kim_vocal_2 inswapper_128; do
  fetch_model models-3.0.0 "$name"
done
fetch_model models-3.1.0 xseg_1
for name in nsfw_1 nsfw_2 nsfw_3; do
  fetch_model models-3.3.0 "$name"
done
```

做第 6.3 节之前下载三个换脸候选；做第 6.5 节之前下载增强模型：

```bash
for name in hyperswap_1a_256 hyperswap_1b_256 hyperswap_1c_256; do
  fetch_model models-3.3.0 "$name"
done
fetch_model models-3.0.0 gfpgan_1.4
```

第 7 节按需下载；不做这些试验就不用下载：

```bash
fetch_model models-3.9.0 hrffa
fetch_model models-3.0.0 peppa_wutz
for name in live_portrait_feature_extractor live_portrait_motion_extractor \
  live_portrait_generator; do
  fetch_model models-3.0.0 "$name"
done
```

下载失败会停止，不会把半个模型作为完成文件。函数不会断点续传，重试会重新下载该文件。关闭终端后若需要再下载，重新执行本节的函数定义，然后只执行尚未完成的 `fetch_model` 命令。

### 3.4 在联网机器下载，拷到另一台机器

可以在联网机器上执行第 3.3 节，再将 **`.assets/models` 整个目录**复制到目标机器同一位置。它是隐藏目录，Finder 按 `Command + Shift + .` 显示隐藏文件。必须同时带上 `.hash`；`.hash` 是 FaceFusion 的 **CRC32** 校验内容，不是你自己生成的 SHA-256 文件。

如果使用 ZIP，解压后的结构应为 `$FF/.assets/models/inswapper_128.onnx`，不能多一层 `models/models/`。完全离线的目标机还要提前完成 Python 依赖安装；模型离线不等于依赖安装也离线。不要把 Linux 的 `.venv` 复制到 Mac。

离线运行前，检查基础组 + 第 6 节需要的权重；只做基础实验时，从 `names` 中删掉 HyperSwap 和 GFPGAN 四项：

```bash
"$FF/.venv/bin/python" - "$FF/.assets/models" <<'PY'
import re
import sys
import zlib
from pathlib import Path
root = Path(sys.argv[1])
names = '''scrfd_2.5g 2dfan4 fan_68_5 arcface_w600k_r50 fairface
bisenet_resnet_34 kim_vocal_2 inswapper_128 xseg_1 nsfw_1 nsfw_2 nsfw_3
hyperswap_1a_256 hyperswap_1b_256 hyperswap_1c_256 gfpgan_1.4'''.split()
for name in names:
    expected = (root / f'{name}.hash').read_text()
    crc = 0
    with (root / f'{name}.onnx').open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            crc = zlib.crc32(block, crc)
    if not re.fullmatch('[0-9a-f]{8}', expected) or f'{crc:08x}' != expected:
        raise SystemExit(f'校验失败：{name}')
    print('OK', name)
PY
```

如果权重有效，本版本会使用本地文件；初始化时仍可能探测下载源，离线环境可能先等待探测超时。联网情况下缺失的所选权重也会自动下载，但本手册先准备完整清单，避免运行中才发现模型遗漏。

## 4. 准备一个失败案例

### 4.1 手里是图片

先修改两个路径。`RAW_ORIGINAL` 必须是**换脸之前的原图**，`RAW_REPLACEMENT` 是你当前使用的替换身份照片。只接受本小节的 JPG/PNG；其他格式先导出为正常显示方向的 JPG/PNG。

```bash
RAW_ORIGINAL="$HOME/Desktop/problem-original.png"     # 修改
RAW_REPLACEMENT="$HOME/Desktop/replacement.jpg"       # 修改
test -f "$RAW_ORIGINAL"
test -f "$RAW_REPLACEMENT"
ffmpeg -hide_banner -loglevel error -n -i "$RAW_ORIGINAL" \
  -frames:v 1 -update 1 "$CASE/original.png"
ffmpeg -hide_banner -loglevel error -n -i "$RAW_REPLACEMENT" \
  -frames:v 1 -update 1 "$CASE/replacement.png"
cp "$CASE/original.png" "$CASE/input/original.png"
cp "$CASE/original.png" "$CASE/reference/reference.png"
cp "$CASE/replacement.png" "$CASE/replacement/replacement.png"
open "$CASE/original.png" "$CASE/replacement.png"
```

这里把问题原图自身用作 media-tool 的匹配参考：对同一图匹配可排除“没有认出该人物”的干扰。**这只是单脸质量诊断，不代表生产环境的跨照片识别已通过验证。** 一张原图中有多张脸时不要照用，要先找单人案例。

查看两张 PNG：应方向正确、五官可见、每张只有一人。替换照片尽量清晰、无重度美颜、无遮挡；先固定当前照片，之后再单独换照片试验，不能换模型同时换身份照片。

### 4.2 手里只有视频

替代第 4.1 节的原图转换：先从**原始视频**取出你不满意的那一帧。修改 `RAW_VIDEO`、时间点 `AT`。输出原图以后，继续第 4.1 节的替换照片转换和三个 `cp` 命令。

```bash
RAW_VIDEO="$HOME/Desktop/original-video.mp4"           # 修改
AT=00:00:12.400                                       # 修改：实际侧脸帧时间
test -f "$RAW_VIDEO"
ffmpeg -hide_banner -loglevel error -n -ss "$AT" -i "$RAW_VIDEO" \
  -frames:v 1 -update 1 "$CASE/original.png"
test -s "$CASE/original.png"
open "$CASE/original.png"
```

如果没有输出，先检查时间点是否超过片长。先完成单图对比，再处理视频；不要第一轮就跑整部影片。

## 5. 定义实验命令：不改原来的配置

以下保存的是本次实验的 Bash 操作命令，不是应用实现。所有输出和配置都写入 `$CASE`。`DET_SIDE=480` 是第一轮固定检测尺寸，**不是宣称 480 总比 640 好**；本地样例出现过替换照片在 480 被检出、在 640 因置信度低于 0.6 被拒绝的情况。先确保两张照片都检出，再独立比较检测尺寸。

```bash
: > "$CASE/empty.ini"
cat > "$CASE/commands.sh" <<'BASH'
export DET_SIDE=480

media_cli() (
  set -e
  # 系统环境变量优先于 .env；只在本次子进程中清除同名覆盖。
  for key in input_material_path work_dir output_material_path source_face \
    target_face models_path detect_method thresholds detection_score_threshold \
    detection_max_side onnx_provider enhance_device enhance_enabled enhance_blend \
    video_encoder video_crf video_preset image_jpeg_quality unmatched_action \
    overwrite on_error cleanup_work_dir log_level; do
    unset "$key"
  done
  cd "$MEDIA"
  "$MEDIA/.venv/bin/python" -m media_tool "$@"
)

media_run() (
  set -e
  set -o pipefail
  local name="$1" enhance="$2"
  test ! -e "$CASE/results/$name"
  mkdir -p "$CASE/results/$name"
  cat > "$CASE/$name.env" <<EOF
input_material_path=$CASE/input
work_dir=$CASE/media-work
output_material_path=$CASE/results/$name
source_face=$CASE/reference
target_face=$CASE/replacement
models_path=$MEDIA/models
detect_method=insightface
thresholds=1.25
detection_score_threshold=0.6
detection_max_side=$DET_SIDE
onnx_provider=cpu
enhance_device=cpu
enhance_enabled=$enhance
enhance_blend=0.7
video_crf=18
video_preset=medium
unmatched_action=copy
overwrite=false
on_error=stop
cleanup_work_dir=true
log_level=DEBUG
EOF
  media_cli check --env "$CASE/$name.env" \
    2>&1 | tee "$CASE/logs/$name-check.log"
  media_cli run --env "$CASE/$name.env" \
    2>&1 | tee "$CASE/logs/$name.log"
  test -s "$CASE/results/$name/original.png"
)

FF_COMMON=(
  --face-detector-model scrfd --face-detector-size "${DET_SIDE}x${DET_SIDE}"
  --face-detector-angles 0 --face-detector-score 0.6
  --face-aligner-model 2dfan4 --face-aligner-score 0
  --face-selector-mode one --face-selector-order large-small
  --face-occluder-model xseg_1 --face-parser-model bisenet_resnet_34
  --voice-extractor-model kim_vocal_2
  --face-tracker-score 0 --face-mask-blur 0.3 --face-mask-padding 0 0 0 0
  --face-swapper-weight 0.5
  --execution-providers cpu --execution-thread-count 4
  --output-image-scale 1.0 --output-image-quality 100
  --log-level debug
)

ff_image() (
  set -e
  set -o pipefail
  local name="$1"
  shift
  test ! -e "$CASE/results/$name"
  cd "$FF"
  "$FF/.venv/bin/python" facefusion.py run \
    --config-path "$CASE/empty.ini" --temp-path "$CASE/ff-temp" \
    --jobs-path "$CASE/ff-jobs" \
    -s "$CASE/replacement.png" -t "$CASE/original.png" \
    -o "$CASE/results/$name" --workflow-mode image-to-image \
    "${FF_COMMON[@]}" "$@" 2>&1 | tee "$CASE/logs/$name.log"
  test -s "$CASE/results/$name"
)
BASH
source "$CASE/commands.sh"
```

说明：`one + large-small` 在 FaceFusion 中选最大的一张脸，因此只用于当前单脸案例。它没有复现 media-tool 的多人参考匹配语义。空 INI 避免旧设置中的年龄/性别等过滤条件影响结果。`face-swapper-weight=0.5` 在此版本映射为零目标身份混入，并不是“只换一半”。

每个实验有独立名称；重试失败的实验时给它一个新名字，例如 `10-ff-ins128-retry.png`。辅助命令会拒绝已有输出，避免你实际查看的是上一次结果。

## 6. 必做实验：照顺序执行

### 6.1 当前流水线与纯换脸

```bash
media_run 01-media-enhance70 true
media_run 02-media-swap-only false
```

查看 `results/01-media-enhance70/original.png` 和 `results/02-media-swap-only/original.png`。

这两张用的是当前实现，但检测尺寸统一为本次实验的 480，CPU 后端固定；不等同于直接沿用你生产 `.env` 中所有参数。若只关闭增强就显著减轻歪脸，先降低/关闭 GFPGAN，不要马上归因于换脸模型。若纯换脸已经五官拉伸，继续检查对齐和模型。

必须查看 `logs/02-media-swap-only.log` 和输出目录的 `.media-tool-reports/*.jsonl`，确认素材记录中的 `matched`、`swapped` 大于 0。未匹配时程序会复制原文件，**命令成功且有图片不代表已经换脸**。

### 6.2 FaceFusion 基线与关键点精修

```bash
ff_image 10-ff-ins128.png --processors face_swapper \
  --face-swapper-model inswapper_128 --face-swapper-pixel-boost 128x128 \
  --face-mask-types box

ff_image 11-ff-ins128-align.png --processors face_swapper \
  --face-swapper-model inswapper_128 --face-swapper-pixel-boost 128x128 \
  --face-aligner-score 0.5 --face-mask-types box

ff_image 12-debug-detector.png --processors face_debugger \
  --face-debugger-items bounding-box face-landmark-5 face-landmark-5/68 face-landmark-68

ff_image 13-debug-align.png --processors face_debugger --face-aligner-score 0.5 \
  --face-debugger-items bounding-box face-landmark-5 face-landmark-5/68 face-landmark-68

ff_image 14-debug-replacement.png --processors face_debugger \
  -t "$CASE/replacement.png" \
  --face-debugger-items bounding-box face-landmark-5 face-landmark-5/68 face-landmark-68
```

`10` 关闭独立关键点精修，`11` 在置信度超过门限时使用 2DFAN4 精修；否则仍会回退。`score=0` 不代表零分点也接受，而是跳过精修分支。两次都会需要基础关键点权重。

比较 `12` 和 `13`：眼睛、鼻尖、嘴角应贴着真实五官，侧脸不可见一侧出现估计点是正常现象，但明显飞到背景或错位值得记录。`14` 检查替换照片是否也被检出。调试图带线条，只用于诊断，不作为质量输出。

对比 `02`、`10` 可以看相似模型在不同流水线中的表现，但不能把差异全部归因于一个环节：检测权重、关键点/遮罩和模型导出也有差别。`10` 与 `11` 才是本节的关键点单因素对照；精修也会影响替换照片的身份提取。

### 6.3 固定对齐和遮罩，比较换脸模型

先确定上一节哪种对齐更好。初始值为 0.5；若 `11` 比 `10` 差，就改为 0。不要在这组中启用增强。

```bash
export ALIGN_SCORE=0.5
ff_image 20-ins128-selected-align.png --processors face_swapper \
  --face-swapper-model inswapper_128 --face-swapper-pixel-boost 128x128 \
  --face-aligner-score "$ALIGN_SCORE" --face-mask-types box

for model in hyperswap_1a_256 hyperswap_1b_256 hyperswap_1c_256; do
  ff_image "21-$model.png" --processors face_swapper \
    --face-swapper-model "$model" --face-swapper-pixel-boost 256x256 \
    --face-aligner-score "$ALIGN_SCORE" --face-mask-types box
done

ff_image 22-ins128-boost256.png --processors face_swapper \
  --face-swapper-model inswapper_128 --face-swapper-pixel-boost 256x256 \
  --face-aligner-score "$ALIGN_SCORE" --face-mask-types box
```

观察顺序：鼻嘴朝向、眼睛数量与位置、脸型是否被拉伸，然后看“像不像替换人物”，最后看清晰度。HyperSwap 三个变体没有对所有人都成立的排名；256 模型不保证侧脸更好。若鼻子方向自然了但身份明显变弱，也要记录。

`22` 使用 INSwapper 的 Pixel Boost，128→256 需要每张脸 4 次换脸推理。它可以作为细节对照，不能代替姿态和几何修复。

选出最好的一张后设置下面两个变量。示例假设 HyperSwap 1a 胜出，**请按你实际比较结果改写**。如果 INSwapper 128 胜出，用 `BEST_MODEL=inswapper_128`、`BEST_BOOST=128x128`；如果 `22` 胜出，使用 256x256。

```bash
export BEST_MODEL=hyperswap_1a_256
export BEST_BOOST=256x256
```

### 6.4 固定模型，只比较遮罩

```bash
ff_image 30-mask-box.png --processors face_swapper \
  --face-swapper-model "$BEST_MODEL" --face-swapper-pixel-boost "$BEST_BOOST" \
  --face-aligner-score "$ALIGN_SCORE" --face-mask-types box

ff_image 31-mask-region.png --processors face_swapper \
  --face-swapper-model "$BEST_MODEL" --face-swapper-pixel-boost "$BEST_BOOST" \
  --face-aligner-score "$ALIGN_SCORE" --face-mask-types box region \
  --face-mask-regions skin left-eyebrow right-eyebrow left-eye right-eye \
    nose mouth upper-lip lower-lip

ff_image 32-mask-occlusion.png --processors face_swapper \
  --face-swapper-model "$BEST_MODEL" --face-swapper-pixel-boost "$BEST_BOOST" \
  --face-aligner-score "$ALIGN_SCORE" --face-mask-types box occlusion

ff_image 33-mask-combined.png --processors face_swapper \
  --face-swapper-model "$BEST_MODEL" --face-swapper-pixel-boost "$BEST_BOOST" \
  --face-aligner-score "$ALIGN_SCORE" --face-mask-types box region occlusion \
  --face-mask-regions skin left-eyebrow right-eyebrow left-eye right-eye \
    nose mouth upper-lip lower-lip
```

`region` 把合成范围限制到解析出的人脸区域，本例排除了 glasses 类；`occlusion` 用于保护部分遮挡区域。检查侧脸外轮廓、头发、耳朵、眼镜、手指是否被错误覆盖，以及是否出现漏换的小块原脸。解析模型也可能判断错；保留得更少并不自动意味着质量更好。

如果只是边缘硬、接缝明显，再单独试 blur 0.15/0.45；不要一开始就把 blur、padding、模型都改了。

按实际胜出的结果设置数组，示例为组合遮罩。若 `30` 胜出，改成 `MASK_TYPES=(box)`；其他同理。

```bash
MASK_TYPES=(box region occlusion)
REGIONS=(skin left-eyebrow right-eyebrow left-eye right-eye nose mouth upper-lip lower-lip)
```

### 6.5 在几何已合格的结果上添加增强

```bash
ff_image 40-best-no-enhancer.png --processors face_swapper \
  --face-swapper-model "$BEST_MODEL" --face-swapper-pixel-boost "$BEST_BOOST" \
  --face-aligner-score "$ALIGN_SCORE" --face-mask-types "${MASK_TYPES[@]}" \
  --face-mask-regions "${REGIONS[@]}"

for blend in 30 50 70; do
  ff_image "41-gfpgan-$blend.png" --processors face_swapper face_enhancer \
    --face-swapper-model "$BEST_MODEL" --face-swapper-pixel-boost "$BEST_BOOST" \
    --face-aligner-score "$ALIGN_SCORE" --face-mask-types "${MASK_TYPES[@]}" \
    --face-mask-regions "${REGIONS[@]}" \
    --face-enhancer-model gfpgan_1.4 --face-enhancer-blend "$blend"
done
open "$CASE/results"
```

30/50/70 是增强像素混合百分比。先从 30 看起：毛孔更自然、眼睛清晰是收益；塑料皮肤、五官重绘、身份变弱是代价。FaceFusion 的增强器会使用 box 和可选 occlusion，**不会原样沿用换脸器的 region 遮罩**，因此添加增强后要重新检查眼镜和轮廓。

如果 `40` 仍然明显歪脸，停止增强试验，返回对齐/换脸模型，不要靠 70 或 100 掩盖问题。

### 6.6 保存本次选型，防止新终端丢失变量

在当前 Bash 中确认变量是你选出的实际值，然后执行：

```bash
declare -p ALIGN_SCORE BEST_MODEL BEST_BOOST MASK_TYPES REGIONS > "$CASE/selection.sh"
```

重新打开终端时，在第 2.4 节后追加 `source "$CASE/selection.sh"`。该文件只保存选择，不自动意味着它们已经通过多案例或视频验收。

## 7. 选做：替换照片、其他关键点、表情

这些试验都从原图重新生成，不以上一张换脸结果继续换脸。先完成第 6 节；本节出现的关键点和表情权重按第 3.3 节下载。

### 7.1 换一张更好的替换照片

用同一个人的清晰、无遮挡、自然表情照片另存为 `$CASE/replacement-alt.png`，然后：

```bash
test -s "$CASE/replacement-alt.png"
ff_image 50-source-photo-alt.png -s "$CASE/replacement-alt.png" \
  --processors face_swapper \
  --face-swapper-model "$BEST_MODEL" --face-swapper-pixel-boost "$BEST_BOOST" \
  --face-aligner-score "$ALIGN_SCORE" --face-mask-types "${MASK_TYPES[@]}" \
  --face-mask-regions "${REGIONS[@]}"
```

和 `40` 比较。先试单张替换照，避免平均多个姿态 embedding 后难以解释变化。不要为了“预处理”把原始侧脸强行拉成正脸；当前 2D 生成器不能可靠恢复被遮住的半张脸。

### 7.2 替换关键点模型

```bash
for aligner in hrffa peppa_wutz; do
  ff_image "51-aligner-$aligner.png" --processors face_swapper \
    --face-swapper-model "$BEST_MODEL" --face-swapper-pixel-boost "$BEST_BOOST" \
    --face-aligner-model "$aligner" --face-aligner-score 0.5 \
    --face-mask-types "${MASK_TYPES[@]}" --face-mask-regions "${REGIONS[@]}"
done
```

与相同配置的 2DFAN4、score 0.5 结果比较；若 `40` 使用了 score 0，需要先补跑一张 2DFAN4、score 0.5 的同配置结果，才能分开判断模型与开关影响。

### 7.3 表情恢复

鼻眼位置已经正常，但张嘴、眨眼、嘴形变化不自然时再试：

```bash
for factor in 30 60; do
  ff_image "52-expression-$factor.png" --processors face_swapper expression_restorer \
    --face-swapper-model "$BEST_MODEL" --face-swapper-pixel-boost "$BEST_BOOST" \
    --face-aligner-score "$ALIGN_SCORE" --face-mask-types "${MASK_TYPES[@]}" \
    --face-mask-regions "${REGIONS[@]}" \
    --expression-restorer-model live_portrait --expression-restorer-factor "$factor"
done
```

和 `40` 比较，暂不混入 GFPGAN。该处理器利用原始目标画面恢复部分表情，不是大侧脸结构修复器；出现重影、嘴部错位就撤掉，不必强行组合所有功能。

## 8. 最后用原始短视频验证

先选择原始视频里包含连续转头的 **2 秒**。`START` 只代表截取开始时间；不要把已有换脸视频放到这里。

```bash
RAW_VIDEO="$HOME/Desktop/original-video.mp4"           # 修改
START=00:00:12.000                                    # 修改
ffmpeg -hide_banner -loglevel error -n -ss "$START" -i "$RAW_VIDEO" -t 2 \
  -map 0:v:0 -map '0:a:0?' -map_chapters -1 \
  -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p \
  -c:a aac -ac 2 -b:a 192k "$CASE/clip-original.mp4"
ffprobe -v error -show_entries format=duration:stream=codec_type,width,height \
  -of json "$CASE/clip-original.mp4"
```

截取命令适用于 SDR。先在播放器确认片段恰好覆盖目标人物和转头过程。为了节省 CPU 时间，这一轮仅保留第一条音轨；不要拿它验证 media-tool 的全部多音轨处理能力。

在当前 Bash 定义并运行两个视频实验，先关闭增强/表情恢复：

```bash
ff_video() (
  set -e
  set -o pipefail
  local name="$1" tracker="$2"
  test ! -e "$CASE/results/$name"
  cd "$FF"
  "$FF/.venv/bin/python" facefusion.py run \
    --config-path "$CASE/empty.ini" --temp-path "$CASE/ff-temp" \
    --jobs-path "$CASE/ff-jobs" \
    -s "$CASE/replacement.png" -t "$CASE/clip-original.mp4" \
    -o "$CASE/results/$name" --workflow-mode image-to-video \
    --workflow-strategy disk --temp-frame-format png \
    "${FF_COMMON[@]}" --processors face_swapper \
    --face-swapper-model "$BEST_MODEL" --face-swapper-pixel-boost "$BEST_BOOST" \
    --face-aligner-score "$ALIGN_SCORE" --face-mask-types "${MASK_TYPES[@]}" \
    --face-mask-regions "${REGIONS[@]}" \
    --target-frame-amount 2 --face-tracker-score "$tracker" \
    --output-video-encoder libx264 --output-video-preset fast \
    --output-video-quality 90 --output-video-scale 1.0 \
    --output-audio-encoder aac 2>&1 | tee "$CASE/logs/$name.log"
  test -s "$CASE/results/$name"
)
ff_video 60-video-no-track.mp4 0
ff_video 61-video-track02.mp4 0.2
open "$CASE/results/60-video-no-track.mp4" "$CASE/results/61-video-track02.mp4"
```

这里两次固定相同的邻帧窗口，只有 tracker score 从 0 改为 0.2。使用 `disk + png` 是为了避开此固定版本默认 `memory` 路径的音轨恢复问题，并避免临时 JPEG 压缩影响比较；两秒片段的临时帧仍会占用磁盘空间。此版本使用框重叠关联和邻帧补全，**不等于完善的关键点时序平滑**，也不保证改善所有转头。多人或切镜时可能关联错，当前只验证单人片段。

原片段有声音时，下面应看到输出的 `audio` 流；不要只凭“processing succeeded”判断音轨保留成功：

```bash
for name in 60-video-no-track.mp4 61-video-track02.mp4; do
  ffprobe -v error -show_entries stream=codec_type,codec_name:format=duration \
    -of json "$CASE/results/$name"
done
```

看正常播放与逐帧两个层面：转头时是否闪烁、忽然回原脸、遮挡区域抖动、身份改变、嘴部重影。无跟踪的结果更好就保持 0。CPU 会较慢，首次加载也耗时；不要用本地 Linux 的速度推断 M1。

需要 media-tool 视频对照时，使用独立视频配置和输出，不把片段塞入前面的单图输入：

```bash
test ! -e "$CASE/results/62-media-video"
mkdir -p "$CASE/video-input"
cp "$CASE/clip-original.mp4" "$CASE/video-input/clip-original.mp4"
"$MEDIA/.venv/bin/python" - "$CASE" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
text = (root / '02-media-swap-only.env').read_text()
text = text.replace(f'input_material_path={root}/input',
                    f'input_material_path={root}/video-input')
text = text.replace(f'output_material_path={root}/results/02-media-swap-only',
                    f'output_material_path={root}/results/62-media-video')
(root / '62-media-video.env').write_text(text)
PY
# 使用第 5 节的命令隔离配置环境变量。
media_cli run --env "$CASE/62-media-video.env" \
  --onnx-provider cpu --enhance-device cpu 2>&1 | tee "$CASE/logs/62-media-video.log"
```

检查视频报告的匹配/换脸数。这里参考集合只有一张侧脸，其他角度可能匹配不上；若出现漏换，先补同一人物的不同角度原始参考照再做独立识别实验，不能把漏换混同为换脸生成质量。

## 9. 如果你确实想“让 FaceFusion 修 media-tool 的结果”

仅在几何基本正确、主要问题是模糊时尝试 **face_enhancer 单处理器**。先用没有增强过的 media-tool 结果，避免无意中叠加两次 GFPGAN：

```bash
ff_image 90-media-plus-ff-enhancer.png \
  -t "$CASE/results/02-media-swap-only/original.png" \
  --processors face_enhancer --face-aligner-score 0.5 \
  --face-mask-types box occlusion \
  --face-enhancer-model gfpgan_1.4 --face-enhancer-blend 30
```

此命令**没有 face_swapper，不会再换身份**。和 `02`、`01` 比较：如果只是更锐利，但鼻嘴仍然歪，就是增强能力的边界。若要试已有的不满意结果，把 `-t` 改成该结果的真实 PNG/JPG 路径，输出名也改成新名字；它仍然是辅助实验，不能作为模型优劣的主要对照。

## 10. 怎么判断下一步，以及常见失败

### 10.1 每个案例填一张记录表

按正常显示比例和 100% 放大各看一次。建议几何、身份、表情、融合各给 1～5 分，5 为最好；另单独记录漏换，避免“没换所以不变形”被当成好结果。

| 输出 | 改变了什么 | 已实际换脸？ | 几何 | 身份 | 表情 | 边缘/遮挡 | 时间 | 保留？ |
|---|---|---|---|---|---|---|---|---|
| 01 / 02 | 当前增强开/关 |  |  |  |  |  |  |  |
| 10 / 11 | 关键点精修开关 |  |  |  |  |  |  |  |
| 20 / 21 三项 | 换脸模型 |  |  |  |  |  |  |  |
| 22 | Pixel Boost |  |  |  |  |  |  |  |
| 30～33 | 遮罩 |  |  |  |  |  |  |  |
| 40 / 41 三项 | 增强强度 |  |  |  |  |  |  |  |
| 60 / 61 | 视频跟踪 |  |  |  |  |  |  |  |

专业建议：不要因一张正脸更锐利就替换默认模型。候选方案至少在左右侧脸、抬头低头、遮挡和短视频中都比基线稳定，且替换身份仍清楚，才值得接入 media-tool。ArcFace 相似度可以补充评估身份，但单一相似度不能证明侧脸几何自然。

| 观察到的结果 | 下一步优先考虑 |
|---|---|
| 关闭增强后歪脸明显减少 | 当前增强重绘过强，先减 blend 或按质量启用 |
| 关键点精修后明显改善 | 借鉴独立关键点、置信度回退，再评估源/目标两侧对齐 |
| HyperSwap 改善几何且身份可接受 | 接入可配置换脸模型及各自输入适配，不只替换文件名 |
| 五官正常，轮廓/眼镜穿帮 | 区域解析与遮挡遮罩更值得优先实现 |
| 单图正常，视频闪烁 | 独立研究跟踪、检测补全、关键点平滑和切镜重置 |
| 大侧脸所有模型仍明显失败 | 记录覆盖边界，研究姿态门控/保留原脸；不要继续叠加增强 |

### 10.2 报错或“看起来没换”时按此排查

| 现象 | 可执行的排查/处理 |
|---|---|
| `invalid choice: headless-run` / 未识别参数 | 本手册用 `run`；检查 `git -C "$FF" rev-parse HEAD` 是否为指定提交 |
| `No module named ...` / NumPy 版本冲突 | 用绝对路径 `.venv/bin/python`；在各自项目按第 2 节安装，不共用环境 |
| Mac 无兼容 ONNX Runtime wheel | 检查系统至少 macOS 14、原生 arm64 Python；不要在 Rosetta x86_64 终端套用 ARM 安装 |
| 缺 `ffmpeg` / `ffprobe` / `curl` | `command -v` 检查；Homebrew 安装后按其提示配置 PATH，再开 Bash |
| 模型校验失败/运行尝试下载 | 检查 `.onnx` 和 `.hash` 成对、路径无多层目录；核对表中标签，再重下对应文件 |
| `choose source image` / 未检出替换脸 | 查看 `14-debug-replacement.png`；换清晰单脸照片。单独试检测尺寸 640，或门限 0.5，不同时改两项 |
| 目标图输出几乎等于原图 | 查看 `12/13` 是否有检测框；media-tool 报告检查匹配/换脸数；成功输出不代表实际换脸 |
| media-tool 提示参考图必须只有一脸 | 检查原图/替换图；重新选择单人案例，不直接放宽人数限制 |
| 输出已存在或实验被跳过 | 换实验名称或新建案例目录；不要复用旧输出判断新参数 |
| 内容分析拒绝处理 | 选择符合项目处理条件的普通素材重新测试；这不是模型下载或换脸参数故障 |
| 视频有文件但漏换/跟错人 | 检查连续原始帧、单脸约束、检测结果及匹配报告，分别排查识别与生成 |
| 原片段有音轨，FaceFusion 输出却无声音 | 确认第 8 节保留了 `--workflow-strategy disk --temp-frame-format png`；用 `ffprobe` 检查音轨，不只看程序退出状态 |

若只想试检测尺寸 640，可以给 FaceFusion 一次实验追加 `--face-detector-size 640x640`。media-tool 则在新的 `media_run` 实验前设置 `DET_SIDE=640`；**已经定义的 `FF_COMMON` 数组不会随变量自动更新**，需重新 `source "$CASE/commands.sh"`（其默认会重置到 480），或显式给 FaceFusion 追加尺寸参数。记下变更，整组对比不要悄悄混用尺寸。

## 11. 命令验证范围与资料

本手册固定版本和模型标签，并核对了源代码中所选模型、共享预检查、下载地址和 CLI 参数。质量是否改善仍取决于你的实际侧脸素材，不承诺 HyperSwap 或增强必然胜出。

验证日期：2026-10-05。验证使用 Linux x86_64 CPU、干净的 FaceFusion 代码副本及按 `requirements.txt` 单独安装的虚拟环境，没有修改两个项目的应用代码。

| 检查 | 实际验证范围 |
|---|---|
| 固定版本与模型下载地址 | 官方提交可访问；全表 21 组 `.onnx` / `.hash` 资源和发布标签已核对 |
| 下载与完整性检查 | 实际下载 `fan_68_5` 并通过 CRC32；media-tool 五项 SHA-256 校验和 FaceFusion 主实验 16 项 CRC32 预检通过 |
| 图片准备 | 图片转 PNG、视频指定时间取帧、两秒 SDR 片段截取命令通过 |
| media-tool 图片 | 增强开/关两个结果通过，均实际匹配并换脸一张 |
| FaceFusion 图片 | 24 次输出通过，覆盖基线、关键点、三种 HyperSwap、Pixel Boost、遮罩、增强、替换照片参数、其他关键点、表情恢复及只增强已有结果 |
| media-tool 视频 | 带音轨合成短片通过，4 帧均实际匹配并换脸 |
| FaceFusion 视频 | 两秒真实片段，磁盘/PNG 路径的跟踪开/关验证；音视频流另用 `ffprobe` 核查 |

macOS Apple Silicon 的 FaceFusion 依赖已按 macOS 14、Python 3.12、ARM64 平台通过纯 wheel 解析检查。**没有在真实 Mac 上运行推理**，Homebrew 安装、图形界面的 `open` 和 Mac 性能需你在目标机确认。Linux 核验时跳过了这些 Mac 专用查看命令。

这些检查证明所列 CLI 的执行路径与模型清单可用，不代表已解决你的严重侧脸失真：当前本地单图样例不能代替你的失败素材，替换照片参数试验也只是验证参数生效。请按第 6 节用自己的案例选型，并在第 8 节确认连续转头的表现。

- [现有质量分析与改造建议](face-swap-quality-improvement.md)
- [media-tool 运行说明](../README.md)及 [模型清单](../models-manifest.json)
- [FaceFusion 固定版本](https://github.com/facefusion/facefusion/tree/9cedad7d2c6f8534f20f66e425d113a655042b07)
- [FaceFusion 模型资源](https://github.com/facefusion/facefusion-assets/releases) / [官方 Hugging Face](https://huggingface.co/facefusion)
