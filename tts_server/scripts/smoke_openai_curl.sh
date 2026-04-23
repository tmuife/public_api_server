#!/usr/bin/env bash
set -euo pipefail

# OpenAI-compatible curl smoke script.
#
# Prerequisites:
# - Service is running locally (default: http://127.0.0.1:8000)
# - curl is installed
#
# Environment variables:
# - OPENAI_BASE_URL: API base URL (default: http://127.0.0.1:8000/v1)
# - OPENAI_API_KEY: Bearer token value (required; falls back to API_KEY)
# - SMOKE_MODEL: model id (default: tts-1)
# - SMOKE_VOICE: voice name (default: alloy)
# - SMOKE_TEXT: synthesis text
# - SMOKE_RESPONSE_FORMAT: wav or pcm (default: wav)
# - SMOKE_STREAM: true or false (default: false)
# - SMOKE_OUTPUT_FILE: output audio file path
#
# Example:
# OPENAI_BASE_URL=http://127.0.0.1:8000/v1 ./scripts/smoke_openai_curl.sh

BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${OPENAI_API_KEY:-${API_KEY:-}}"
MODEL="${SMOKE_MODEL:-tts-1}"
VOICE="${SMOKE_VOICE:-alloy}"
TEXT="${SMOKE_TEXT:-curl smoke request from tts-server}"
RESPONSE_FORMAT="${SMOKE_RESPONSE_FORMAT:-wav}"
STREAM="${SMOKE_STREAM:-false}"
OUTPUT_FILE="${SMOKE_OUTPUT_FILE:-/tmp/tts-smoke-${RESPONSE_FORMAT}.bin}"

if [[ "${RESPONSE_FORMAT}" != "wav" && "${RESPONSE_FORMAT}" != "pcm" ]]; then
  echo "[FAIL] SMOKE_RESPONSE_FORMAT must be wav or pcm, got: ${RESPONSE_FORMAT}" >&2
  exit 1
fi

if [[ "${STREAM}" != "true" && "${STREAM}" != "false" ]]; then
  echo "[FAIL] SMOKE_STREAM must be true or false, got: ${STREAM}" >&2
  exit 1
fi

if [[ -z "${API_KEY}" ]]; then
  echo "[FAIL] Missing OPENAI_API_KEY (or API_KEY). Set it before running smoke." >&2
  exit 1
fi

TMP_BODY="$(mktemp)"
trap 'rm -f "${TMP_BODY}"' EXIT

HTTP_STATUS="$(
  curl \
    --silent \
    --show-error \
    --location \
    --request POST \
    --url "${BASE_URL%/}/audio/speech" \
    --header "Authorization: Bearer ${API_KEY}" \
    --header "Content-Type: application/json" \
    --data "$(cat <<JSON
{"model":"${MODEL}","input":"${TEXT}","voice":"${VOICE}","response_format":"${RESPONSE_FORMAT}","stream":${STREAM}}
JSON
)" \
    --output "${TMP_BODY}" \
    --write-out "%{http_code}"
)"

if [[ "${HTTP_STATUS}" != "200" ]]; then
  echo "[FAIL] /v1/audio/speech returned HTTP ${HTTP_STATUS}" >&2
  if [[ -s "${TMP_BODY}" ]]; then
    echo "Response body:"
    cat "${TMP_BODY}"
  fi
  exit 1
fi

if [[ ! -s "${TMP_BODY}" ]]; then
  echo "[FAIL] /v1/audio/speech returned empty body" >&2
  exit 1
fi

mv "${TMP_BODY}" "${OUTPUT_FILE}"
BYTES="$(wc -c < "${OUTPUT_FILE}")"
echo "[PASS] curl smoke succeeded (${BYTES} bytes). Output: ${OUTPUT_FILE}"
