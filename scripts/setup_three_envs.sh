#!/usr/bin/env bash
# Create isolated Pi3X, MoGe-3, and VIPE Python environments and install each model.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3.10}"
ENV_ROOT="${ENV_ROOT:-${PROJECT_ROOT}/.envs}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/third_party}"
PI3_TORCH_INDEX_URL="${PI3_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
MOGE_TORCH_INDEX_URL="${MOGE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
VIPE_TORCH_INDEX_URL="${VIPE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ ! -d "${SOURCE_ROOT}/Pi3/.git" || ! -d "${SOURCE_ROOT}/MoGe/.git" || ! -d "${SOURCE_ROOT}/vipe/.git" ]]; then
  echo "Missing upstream source. Run scripts/clone_models.sh first." >&2
  exit 2
fi

create_env() {
  local path="$1"
  "${PYTHON_COMMAND}" -m venv "${path}"
  "${path}/bin/python" -m pip install --upgrade pip setuptools wheel
}

mkdir -p "${ENV_ROOT}"
create_env "${ENV_ROOT}/pi3x"
create_env "${ENV_ROOT}/moge3"
create_env "${ENV_ROOT}/vipe"

# Pi3's official requirements pin torch 2.5.1, torchvision 0.20.1, NumPy 1.26.4.
"${ENV_ROOT}/pi3x/bin/python" -m pip install \
  torch==2.5.1 torchvision==0.20.1 --index-url "${PI3_TORCH_INDEX_URL}"
"${ENV_ROOT}/pi3x/bin/python" -m pip install -r "${SOURCE_ROOT}/Pi3/requirements.txt"
"${ENV_ROOT}/pi3x/bin/python" -m pip install -e "${SOURCE_ROOT}/Pi3"
"${ENV_ROOT}/pi3x/bin/python" -m pip install --no-deps -e "${PROJECT_ROOT}"
"${ENV_ROOT}/pi3x/bin/python" -m pip install scipy tqdm

# MoGe-3 requires NumPy 2.x and adds Triton/FlexGEMM-based sparse refinement.
"${ENV_ROOT}/moge3/bin/python" -m pip install \
  torch torchvision --index-url "${MOGE_TORCH_INDEX_URL}"
"${ENV_ROOT}/moge3/bin/python" -m pip install -e "${SOURCE_ROOT}/MoGe"

# VIPE builds a CUDA extension during installation; nvcc and CUDA-enabled Torch are required.
"${ENV_ROOT}/vipe/bin/python" -m pip install \
  torch torchvision --index-url "${VIPE_TORCH_INDEX_URL}"
"${ENV_ROOT}/vipe/bin/python" "${PROJECT_ROOT}/scripts/setup_vipe.py" \
  --vipe-source "${SOURCE_ROOT}/vipe"

mkdir -p "${PROJECT_ROOT}/ckpt/pi3x" "${PROJECT_ROOT}/ckpt/moge3" "${PROJECT_ROOT}/ckpt/vipe"
"${ENV_ROOT}/pi3x/bin/python" "${PROJECT_ROOT}/scripts/check_three_envs.py" \
  --env-root "${ENV_ROOT}" --project-root "${PROJECT_ROOT}" --skip-checkpoints

echo "Three environments created under ${ENV_ROOT}"
