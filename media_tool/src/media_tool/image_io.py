from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps
from pillow_heif import register_heif_opener

from .errors import MediaToolError

register_heif_opener()
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".gif",
    ".avif",
}


@dataclass
class ImageData:
    frame: np.ndarray
    alpha: np.ndarray | None


def read_image(path: Path) -> ImageData:
    try:
        with Image.open(path) as opened:
            if getattr(opened, "n_frames", 1) != 1 or opened.format in {"GIF", "AVIF"}:
                raise MediaToolError(f"首版不支持动画、多页、GIF 或 AVIF 图片：{path}")
            if opened.mode in {"I", "F", "I;16", "I;16B"} or opened.info.get("bit_depth", 8) > 8:
                raise MediaToolError(f"首版只支持 8 位 SDR 静态图片：{path}")
            oriented = ImageOps.exif_transpose(opened)
            alpha = (
                np.array(oriented.convert("RGBA"))[:, :, 3]
                if ("A" in oriented.getbands() or "transparency" in oriented.info)
                else None
            )
            rgb = oriented.convert("RGB")
            profile = opened.info.get("icc_profile")
            if profile:
                source = ImageCms.ImageCmsProfile(BytesIO(profile))
                color_input = oriented if oriented.mode == "CMYK" else rgb
                rgb = ImageCms.profileToProfile(
                    color_input, source, ImageCms.createProfile("sRGB"), outputMode="RGB"
                )
            frame = np.array(rgb)[:, :, ::-1].copy()
            return ImageData(frame, alpha)
    except MediaToolError:
        raise
    except Exception as exc:
        raise MediaToolError(f"图片解码失败 {path}: {exc}") from exc


def write_image(path: Path, image: ImageData, quality: int) -> None:
    output = Image.fromarray(image.frame[:, :, ::-1])
    if image.alpha is not None and path.suffix.lower() in {".png", ".webp", ".tif", ".tiff"}:
        output.putalpha(Image.fromarray(image.alpha))
    formats = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".webp": "WEBP",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
    }
    image_format = formats.get(path.suffix.lower())
    if image_format is None:
        raise MediaToolError(f"不支持的输出图片格式：{path.suffix}")
    options = {"quality": quality} if image_format in {"JPEG", "WEBP"} else {}
    if image_format == "JPEG":
        options["subsampling"] = 0
    output.save(path, format=image_format, **options)
    with Image.open(path) as check:
        check.load()
        if check.size != (image.frame.shape[1], image.frame.shape[0]):
            raise MediaToolError("输出图片尺寸校验失败")
