#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${COMFY_ROOT:-/opt/ComfyUI}"
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_VOLUME_ROOT="${COMFY_VOLUME_ROOT:-/workspace/ComfyUI}"
COMFY_MODELS_DIR="${COMFY_MODELS_DIR:-${COMFY_VOLUME_ROOT}/models}"
COMFY_OUTPUT_DIR="${COMFY_OUTPUT_DIR:-${COMFY_VOLUME_ROOT}/output}"
COMFY_INPUT_DIR="${COMFY_INPUT_DIR:-${COMFY_VOLUME_ROOT}/input}"
COMFY_USER_DIR="${COMFY_USER_DIR:-${COMFY_VOLUME_ROOT}/user}"
COMFY_CUSTOM_NODES_DIR="${COMFY_ROOT}/custom_nodes"

mkdir -p "${COMFY_MODELS_DIR}" "${COMFY_OUTPUT_DIR}" "${COMFY_INPUT_DIR}" "${COMFY_USER_DIR}"

rm -rf "${COMFY_ROOT}/models" "${COMFY_ROOT}/output" "${COMFY_ROOT}/input" "${COMFY_ROOT}/user"

ln -sfn "${COMFY_MODELS_DIR}" "${COMFY_ROOT}/models"
ln -sfn "${COMFY_OUTPUT_DIR}" "${COMFY_ROOT}/output"
ln -sfn "${COMFY_INPUT_DIR}" "${COMFY_ROOT}/input"
ln -sfn "${COMFY_USER_DIR}" "${COMFY_ROOT}/user"

python "${COMFY_ROOT}/main.py" \
  --listen 127.0.0.1 \
  --port "${COMFY_PORT}" \
  --disable-auto-launch \
  --reserve-vram 5 &

COMFY_PID=$!

cleanup() {
  kill "${COMFY_PID}" 2>/dev/null || true
}

trap cleanup EXIT

python -u /app/handler.py
