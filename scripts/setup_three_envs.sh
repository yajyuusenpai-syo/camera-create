#!/usr/bin/env bash
# Create isolated Pi3X, MoGe-3, and VIPE Python environments and install each model.
set -euo pipefail

# Prevent system/user Python packages from leaking into model virtual environments.
unset PYTHONPATH
unset PYTHONHOME
unset PIP_USER
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3.10}"
ENV_ROOT="${ENV_ROOT:-${PROJECT_ROOT}/.envs}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/third_party}"
PI3_TORCH_INDEX_URL="${PI3_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
MOGE_TORCH_INDEX_URL="${MOGE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
VIPE_TORCH_INDEX_URL="${VIPE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
PI3_WHEELHOUSE="${PI3_WHEELHOUSE:-}"
MOGE_WHEELHOUSE="${MOGE_WHEELHOUSE:-}"
VIPE_WHEELHOUSE="${VIPE_WHEELHOUSE:-}"

if [[ ! -d "${SOURCE_ROOT}/Pi3/.git" || ! -d "${SOURCE_ROOT}/MoGe/.git" || ! -d "${SOURCE_ROOT}/vipe/.git" ]]; then
  echo "Missing upstream source. Run scripts/clone_models.sh first." >&2
  exit 2
fi

create_env() {
  local path="$1"
  "${PYTHON_COMMAND}" -m venv "${path}"
  "${path}/bin/python" -m pip install --upgrade pip setuptools wheel
}

install_torch() {
  local python="$1"
  local cuda_index="$2"
  local wheelhouse="$3"
  shift 3
  if [[ -n "${wheelhouse}" ]]; then
    if [[ ! -d "${wheelhouse}" ]]; then
      echo "Torch wheelhouse does not exist: ${wheelhouse}" >&2
      exit 2
    fi
    "${python}" -m pip install --no-index --find-links "${wheelhouse}" "$@"
  else
    "${python}" -m pip install "$@" \
      --index-url "${cuda_index}" --extra-index-url "${PYPI_INDEX_URL}"
  fi
}

mkdir -p "${ENV_ROOT}"
create_env "${ENV_ROOT}/pi3x"
create_env "${ENV_ROOT}/moge3"
create_env "${ENV_ROOT}/vipe"

# Pi3's official requirements pin torch 2.5.1, torchvision 0.20.1, NumPy 1.26.4.
install_torch "${ENV_ROOT}/pi3x/bin/python" "${PI3_TORCH_INDEX_URL}" \
  "${PI3_WHEELHOUSE}" torch==2.5.1 torchvision==0.20.1
"${ENV_ROOT}/pi3x/bin/python" -m pip install -r "${SOURCE_ROOT}/Pi3/requirements.txt"
"${ENV_ROOT}/pi3x/bin/python" -m pip install -e "${SOURCE_ROOT}/Pi3"
"${ENV_ROOT}/pi3x/bin/python" -m pip install --no-deps -e "${PROJECT_ROOT}"
"${ENV_ROOT}/pi3x/bin/python" -m pip install scipy tqdm

# MoGe-3 requires NumPy 2.x and adds Triton/FlexGEMM-based sparse refinement.
install_torch "${ENV_ROOT}/moge3/bin/python" "${MOGE_TORCH_INDEX_URL}" \
  "${MOGE_WHEELHOUSE}" torch torchvision
"${ENV_ROOT}/moge3/bin/python" -m pip install -e "${SOURCE_ROOT}/MoGe"

# VIPE builds a CUDA extension during installation; nvcc and CUDA-enabled Torch are required.
install_torch "${ENV_ROOT}/vipe/bin/python" "${VIPE_TORCH_INDEX_URL}" \
  "${VIPE_WHEELHOUSE}" torch torchvision
"${ENV_ROOT}/vipe/bin/python" "${PROJECT_ROOT}/scripts/setup_vipe.py" \
  --vipe-source "${SOURCE_ROOT}/vipe"

mkdir -p "${PROJECT_ROOT}/ckpt/pi3x" "${PROJECT_ROOT}/ckpt/moge3" "${PROJECT_ROOT}/ckpt/vipe"
"${ENV_ROOT}/pi3x/bin/python" "${PROJECT_ROOT}/scripts/check_three_envs.py" \
  --env-root "${ENV_ROOT}" --project-root "${PROJECT_ROOT}" --skip-checkpoints

echo "Three environments created under ${ENV_ROOT}"
