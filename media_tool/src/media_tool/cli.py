import argparse
import json
import logging
from pathlib import Path
import sys
import time

from .config import load_settings
from .errors import MediaToolError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="独立的图片、视频匹配换脸与人脸增强工具")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        command = subcommands.add_parser(
            name, help="检查环境与模型" if name == "check" else "批处理素材"
        )
        command.add_argument("--env", type=Path, default=Path(".env"), help="配置文件路径")
        command.add_argument("--detect-method", choices=["insightface", "yunet"])
        command.add_argument("--onnx-provider", choices=["auto", "cpu", "coreml"])
        command.add_argument("--enhance-device", choices=["auto", "cpu", "mps"])
        if name == "run":
            command.add_argument("--dry-run", action="store_true", help="只检查清单与输出映射")
    arguments = parser.parse_args(argv)
    try:
        overrides = {
            name: getattr(arguments, name)
            for name in ("detect_method", "onnx_provider", "enhance_device")
            if getattr(arguments, name) is not None
        }
        settings = load_settings(arguments.env, overrides)
        logging.basicConfig(
            level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s"
        )
        from .video_io import check_tools

        versions = check_tools()
        settings.required_models()
        if arguments.command == "run" and arguments.dry_run:
            from .runner import dry_run

            items = dry_run(settings)
            print(
                json.dumps(
                    {"detect_method": settings.detect_method, "materials": items},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1 if any("error" in item for item in items) else 0
        # 控制 CPU 推理线程；关闭增强时不导入 PyTorch。
        if settings.enhance_enabled:
            import torch

            torch.set_num_threads(4)
        from .pipeline import build_pipeline

        startup = time.perf_counter()
        pipeline = build_pipeline(settings)
        if arguments.command == "check":
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "tools": versions,
                        "detect_method": settings.detect_method,
                        "references": len(pipeline.matcher.names),
                        "backends": pipeline.backend_info(),
                        "startup_seconds": round(time.perf_counter() - startup, 3),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        from .runner import run_batch

        code, report = run_batch(settings, pipeline)
        print(f"运行报告：{report}")
        return code
    except KeyboardInterrupt:
        print("已中断，当前半成品不会发布。", file=sys.stderr)
        return 130
    except (MediaToolError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
