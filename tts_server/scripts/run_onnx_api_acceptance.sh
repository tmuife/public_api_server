#!/usr/bin/env bash
set -euo pipefail

# End-to-end acceptance entrypoint for ONNX OpenAI-compatible API checks.
# This workflow runs:
# 1) API acceptance tests (stream/batch + core negative cases)
# 2) Torchless startup verification
# 3) Reference-directory independence smoke validation

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[1/3] Running ONNX API acceptance tests"
uv run python -m unittest tests.test_onnx_api_acceptance -v

echo "[2/3] Running torchless startup validation"
uv run python -m unittest tests.test_torchless_startup -v

echo "[3/3] Running reference-directory independence smoke validation"
uv run python scripts/smoke_without_reference_dirs.py

echo "ONNX API acceptance workflow completed successfully."
