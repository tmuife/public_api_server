from dataclasses import dataclass
import os
from pathlib import Path
import unicodedata

from .errors import MediaToolError
from .image_io import IMAGE_EXTENSIONS

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".ts",
    ".mts",
    ".m2ts",
    ".wmv",
    ".flv",
}


def scan_files(root: Path, extensions: set[str]) -> list[Path]:
    result = []
    for folder, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name for name in directories if not (Path(folder) / name).is_symlink()
        )
        for name in sorted(filenames):
            path = Path(folder) / name
            if not path.is_symlink() and path.is_file() and path.suffix.lower() in extensions:
                result.append(path)
    return sorted(result, key=lambda path: str(path.relative_to(root)))


@dataclass(frozen=True)
class Material:
    source: Path
    processed: Path
    original: Path
    kind: str


def plan_materials(settings) -> list[Material]:
    result, owners = [], {}
    files = scan_files(settings.input_material_path, IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    for source in files:
        relative = source.relative_to(settings.input_material_path)
        if relative.parts[0].casefold() == ".media-tool-reports":
            raise MediaToolError("输入素材不能使用保留目录名 .media-tool-reports")
        original = settings.output_material_path / relative
        kind = "image" if source.suffix.lower() in IMAGE_EXTENSIONS else "video"
        processed = original
        if kind == "video" and source.suffix.lower() != ".mp4":
            processed = original.with_name(original.name + ".mp4")
        if kind == "image" and source.suffix.lower() in {".heic", ".heif"}:
            processed = original.with_name(original.name + ".jpg")
        for target in {original, processed}:
            key = unicodedata.normalize("NFC", str(target)).casefold()
            if key in owners and owners[key] != source:
                raise MediaToolError(f"输出路径冲突：{owners[key]} / {source} -> {target}")
            owners[key] = source
            # 防止已有输出子目录符号链接将最终文件导向输出根目录之外。
            if not target.resolve().is_relative_to(settings.output_material_path):
                raise MediaToolError(f"输出路径经过符号链接离开输出目录：{target}")
            if target.exists() and not target.is_file():
                raise MediaToolError(f"输出候选路径不是文件：{target}")
        result.append(Material(source, processed, original, kind))
    return result
