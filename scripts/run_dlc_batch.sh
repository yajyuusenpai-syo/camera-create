#!/usr/bin/env bash
# Launch one independent manifest shard with 8 GPUs and fused SDP/cuDNN disabled.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 INPUT_MANIFEST [CHECKPOINT_DIR] [RUN_ID]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
INPUT_MANIFEST="$1"
CHECKPOINT_DIR="${2:-}"
RUN_ID="${3:-}"

PYTHON_BIN="${CAMERA_CREATE_PYTHON:-${PROJECT_ROOT}/.envs/pi3x/bin/python}"
EXTRA_ARGS=()
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  EXTRA_ARGS+=(--checkpoint-dir "${CHECKPOINT_DIR}")
fi
if [[ -n "${RUN_ID}" ]]; then
  EXTRA_ARGS+=(--run-id "${RUN_ID}")
fi
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/cli.py" \
  --input "${INPUT_MANIFEST}" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 4 \
  --depth-services-per-gpu 4 \
  --disable-cudnn \
  --disable-sdp \
  "${EXTRA_ARGS[@]}"
