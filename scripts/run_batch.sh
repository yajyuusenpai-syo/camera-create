#!/usr/bin/env bash
# Launch recursive multi-GPU camera extraction with the project Pi3X controller Python.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROLLER_PYTHON="${CONTROLLER_PYTHON:-${PROJECT_ROOT}/.envs/pi3x/bin/python}"

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_batch.sh INPUT_DIRECTORY [camera-create options]" >&2
  exit 2
fi

INPUT_DIRECTORY="$1"
shift
exec "${CONTROLLER_PYTHON}" "${PROJECT_ROOT}/cli.py" \
  --input "${INPUT_DIRECTORY}" "$@"
