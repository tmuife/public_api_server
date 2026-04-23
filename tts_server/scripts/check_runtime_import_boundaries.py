from __future__ import annotations

import ast
import sys
from pathlib import Path

FORBIDDEN_SOURCE_SNIPPETS = (
    "MOSS-TTS-Nano-main",
    "Kokoro-FastAPI-master",
)
FORBIDDEN_MODULES = {
    "torch",
    "torchaudio",
}


def iter_python_files() -> list[Path]:
    files = [Path("main.py")]
    files.extend(sorted(Path("app").rglob("*.py")))
    return files


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    source = path.read_text(encoding="utf-8")

    for snippet in FORBIDDEN_SOURCE_SNIPPETS:
        if snippet in source:
            errors.append(f"{path}: contains forbidden reference snippet '{snippet}'")

    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    errors.append(f"{path}:{node.lineno} imports forbidden module '{alias.name}'")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_MODULES:
                errors.append(f"{path}:{node.lineno} imports forbidden module '{node.module}'")

    return errors


def main() -> int:
    all_errors: list[str] = []
    for file_path in iter_python_files():
        if not file_path.exists():
            continue
        all_errors.extend(check_file(file_path))

    if all_errors:
        print("Import boundary validation failed:")
        for error in all_errors:
            print(f"- {error}")
        return 1

    print("Import boundary validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
