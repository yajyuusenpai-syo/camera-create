#!/usr/bin/env bash
# Create a Linux Python virtual environment and install camera_create in editable mode.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3.10}"
ENVIRONMENT_PATH="${ENVIRONMENT_PATH:-${PROJECT_ROOT}/.venv}"

"${PYTHON_COMMAND}" -m venv "${ENVIRONMENT_PATH}"
"${ENVIRONMENT_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENVIRONMENT_PATH}/bin/python" -m pip install -e "${PROJECT_ROOT}[dev]"
echo "Environment ready. Activate with: source ${ENVIRONMENT_PATH}/bin/activate"

