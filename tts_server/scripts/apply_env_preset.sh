#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <natural|stable>" >&2
  exit 1
fi

PRESET_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRESET_PATH="${PROJECT_ROOT}/.env.preset.${PRESET_NAME}"
TARGET_PATH="${PROJECT_ROOT}/.env"

if [[ ! -f "${PRESET_PATH}" ]]; then
  echo "Preset not found: ${PRESET_PATH}" >&2
  exit 1
fi

cp "${PRESET_PATH}" "${TARGET_PATH}"
echo "Applied preset '${PRESET_NAME}' to ${TARGET_PATH}"
