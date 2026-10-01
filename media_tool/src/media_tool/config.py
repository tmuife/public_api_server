from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path

from dotenv import dotenv_values

from .errors import MediaToolError


@dataclass(frozen=True)
class Settings:
    input_material_path: Path
    work_dir: Path
    output_material_path: Path
    source_face: Path
    target_face: Path
    models_path: Path
    detect_method: str = "insightface"
    thresholds: float = 1.25
    detection_score_threshold: float = 0.6
    detection_max_side: int = 640
    onnx_provider: str = "auto"
    enhance_device: str = "auto"
    enhance_enabled: bool = True
    enhance_blend: float = 0.7
    video_encoder: str = "libx264"
    video_crf: int = 18
    video_preset: str = "medium"
    image_jpeg_quality: int = 95
    unmatched_action: str = "copy"
    overwrite: bool = False
    on_error: str = "continue"
    cleanup_work_dir: bool = True
    log_level: str = "INFO"

    def validate(self) -> None:
        enums = {
            "detect_method": {"insightface", "yunet"},
            "onnx_provider": {"cpu", "coreml", "auto"},
            "enhance_device": {"cpu", "mps", "auto"},
            "video_encoder": {"libx264"},
            "video_preset": {
                "ultrafast",
                "superfast",
                "veryfast",
                "faster",
                "fast",
                "medium",
                "slow",
                "slower",
                "veryslow",
            },
            "unmatched_action": {"copy", "skip"},
            "on_error": {"continue", "stop"},
            "log_level": {"DEBUG", "INFO", "WARNING", "ERROR"},
        }
        for name, choices in enums.items():
            if getattr(self, name) not in choices:
                raise MediaToolError(f"{name} 必须是 {', '.join(sorted(choices))}")
        for name, low, high, inclusive in [
            ("thresholds", 0, 2, False),
            ("detection_score_threshold", 0, 1, False),
            ("enhance_blend", 0, 1, True),
        ]:
            value = getattr(self, name)
            if (
                not math.isfinite(value)
                or value > high
                or (value < low if inclusive else value <= low)
            ):
                raise MediaToolError(f"{name} 超出允许范围")
        if self.detection_max_side < 32 or self.detection_max_side % 32:
            raise MediaToolError("detection_max_side 必须是至少 32 的 32 倍数")
        if not 0 <= self.video_crf <= 51 or not 1 <= self.image_jpeg_quality <= 100:
            raise MediaToolError("video_crf 必须为 0..51；image_jpeg_quality 必须为 1..100")
        roots = [self.input_material_path, self.work_dir, self.output_material_path]
        for i, left in enumerate(roots):
            for right in roots[i + 1 :]:
                if left.is_relative_to(right) or right.is_relative_to(left):
                    raise MediaToolError(
                        f"输入、工作、输出目录不能相同或互相包含：{left} / {right}"
                    )
        for name in ("source_face", "target_face", "models_path"):
            if getattr(self, name).is_relative_to(self.input_material_path):
                raise MediaToolError(f"{name} 不能位于输入目录中")
        # 工作根目录的清理范围必须与模型、参考图、替换图隔离。
        for root in (self.work_dir, self.output_material_path):
            for name in ("source_face", "target_face", "models_path"):
                other = getattr(self, name)
                if root.is_relative_to(other) or other.is_relative_to(root):
                    raise MediaToolError(f"工作/输出目录不能与 {name} 互相包含")
        for name in ("input_material_path", "source_face", "target_face", "models_path"):
            if not getattr(self, name).is_dir():
                raise MediaToolError(f"{name} 目录不存在：{getattr(self, name)}")

    def required_models(self) -> dict[str, Path]:
        detector = (
            "detection/insightface/det_10g.onnx"
            if self.detect_method == "insightface"
            else "detection/yunet/face_detection_yunet_2023mar.onnx"
        )
        paths = {
            "detector": detector,
            "recognition": "recognition/w600k_r50.onnx",
            "swap": "swap/inswapper_128.onnx",
        }
        if self.enhance_enabled:
            paths["enhance"] = "enhance/GFPGANv1.4.pth"
        resolved = {key: self.models_path / value for key, value in paths.items()}
        missing = [str(path) for path in resolved.values() if not path.is_file()]
        if missing:
            raise MediaToolError("缺少本地模型（不会自动下载）：\n" + "\n".join(missing))
        return resolved

    def report_dict(self) -> dict:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }


def load_settings(env_file: Path, overrides: dict | None = None) -> Settings:
    env_file = env_file.expanduser().resolve()
    if not env_file.is_file():
        raise MediaToolError(f"配置文件不存在：{env_file}；请复制 .env.example 后填写路径")
    raw = dict(dotenv_values(env_file, interpolate=False))
    fields = Settings.__dataclass_fields__
    unknown = set(raw) - set(fields)
    if unknown:
        raise MediaToolError("未知配置项：" + ", ".join(sorted(unknown)))
    for name in fields:
        if name in os.environ:
            raw[name] = os.environ[name]
    raw.update(overrides or {})
    path_names = {
        "input_material_path",
        "work_dir",
        "output_material_path",
        "source_face",
        "target_face",
        "models_path",
    }
    bool_names = {"enhance_enabled", "overwrite", "cleanup_work_dir"}
    float_names = {"thresholds", "detection_score_threshold", "enhance_blend"}
    int_names = {"detection_max_side", "video_crf", "image_jpeg_quality"}
    values = {}
    for name, value in raw.items():
        if value is None or not str(value).strip():
            raise MediaToolError(f"配置项 {name} 不能为空")
        value = str(value).strip()
        try:
            if name in path_names:
                path = Path(value).expanduser()
                values[name] = (path if path.is_absolute() else env_file.parent / path).resolve()
            elif name in bool_names:
                if value.lower() not in {"true", "false"}:
                    raise ValueError("必须为 true 或 false")
                values[name] = value.lower() == "true"
            elif name in float_names:
                values[name] = float(value)
            elif name in int_names:
                values[name] = int(value)
            else:
                values[name] = value.upper() if name == "log_level" else value
        except ValueError as exc:
            raise MediaToolError(f"配置项 {name} 格式不正确：{exc}") from exc
    missing = path_names - values.keys()
    if missing:
        raise MediaToolError("缺少配置路径：" + ", ".join(sorted(missing)))
    settings = Settings(**values)
    settings.validate()
    return settings
