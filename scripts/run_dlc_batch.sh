#!/usr/bin/env bash
# Launch one safe DLC camera batch shard with 8 GPUs, 4 workers/GPU and fused SDP/cuDNN disabled.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 INPUT_DIR SHARED_CHECKPOINT_DIR [RUN_ID]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
INPUT_DIR="$1"
CHECKPOINT_DIR="$2"
RUN_ID="${3:-${CAMERA_CREATE_RUN_ID:-${DLC_JOB_ID:-${PAI_JOB_ID:-}}}}"

if [[ -z "${RUN_ID}" ]]; then
  echo "RUN_ID is required: pass it as argument 3 or set CAMERA_CREATE_RUN_ID/DLC_JOB_ID." >&2
  exit 2
fi

WORLD="${WORLD_SIZE:?DLC must provide WORLD_SIZE}"
GLOBAL_RANK="${RANK:?DLC must provide RANK}"
MASTER_HOST="${MASTER_ADDR:?DLC must provide MASTER_ADDR}"
MASTER_SERVICE_PORT="${MASTER_PORT:?DLC must provide MASTER_PORT}"
LOCAL_WORLD="${LOCAL_WORLD_SIZE:-1}"
if [[ -n "${GROUP_WORLD_SIZE:-}" && -n "${GROUP_RANK:-}" ]]; then
  MACHINE_COUNT="${GROUP_WORLD_SIZE}"
  MACHINE_RANK="${GROUP_RANK}"
elif (( LOCAL_WORLD > 1 )); then
  if (( WORLD % LOCAL_WORLD != 0 )); then
    echo "WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE." >&2
    exit 2
  fi
  MACHINE_COUNT="$((WORLD / LOCAL_WORLD))"
  MACHINE_RANK="$((GLOBAL_RANK / LOCAL_WORLD))"
else
  MACHINE_COUNT="${WORLD}"
  MACHINE_RANK="${GLOBAL_RANK}"
fi

PYTHON_BIN="${CAMERA_CREATE_PYTHON:-${PROJECT_ROOT}/.envs/pi3x/bin/python}"
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/cli.py" \
  --input "${INPUT_DIR}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --run-id "${RUN_ID}" \
  --num_machines "${MACHINE_COUNT}" \
  --num_processes "$((MACHINE_COUNT * 8))" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MASTER_HOST}" \
  --main_process_port "${MASTER_SERVICE_PORT}" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 4 \
  --disable-cudnn \
  --disable-sdp
