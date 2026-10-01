"""兼容 uv run python main.py；推荐使用 uv run media-tool。"""

from media_tool.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
