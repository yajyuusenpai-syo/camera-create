#!/usr/bin/env bash
# Create isolated Pi3X, MoGe-3, and VIPE Conda-prefix environments for restricted servers.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_COMMAND="${CONDA_COMMAND:-conda}"
ENV_ROOT="${ENV_ROOT:-${PROJECT_ROOT}/.envs}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/third_party}"
PI3_TORCH_INDEX_URL="${PI3_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
MOGE_TORCH_INDEX_URL="${MOGE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
VIPE_TORCH_INDEX_URL="${VIPE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ ! -d "${SOURCE_ROOT}/Pi3/.git" || ! -d "${SOURCE_ROOT}/MoGe/.git" || ! -d "${SOURCE_ROOT}/vipe/.git" ]]; then
  echo "Missing upstream source. Run scripts/clone_models.sh first." >&2
  exit 2
fi

if ! command -v "${CONDA_COMMAND}" >/dev/null 2>&1; then
  echo "Conda executable not found: ${CONDA_COMMAND}" >&2
  exit 2
fi

create_env() {
  local path="$1"
  if [[ -e "${path}" && ! -f "${path}/conda-meta/history" ]]; then
    echo "Refusing to reuse non-Conda environment path: ${path}" >&2
    exit 2
  fi
  "${CONDA_COMMAND}" create --prefix "${path}" python=3.10 pip -y
  "${CONDA_COMMAND}" run --prefix "${path}" python -m pip install --upgrade pip setuptools wheel
}

run_python() {
  local path="$1"
  shift
  "${CONDA_COMMAND}" run --prefix "${path}" python "$@"
}

mkdir -p "${ENV_ROOT}"
create_env "${ENV_ROOT}/pi3x"
create_env "${ENV_ROOT}/moge3"
create_env "${ENV_ROOT}/vipe"

# Pi3X keeps its official Torch/NumPy pins inside a dedicated Conda prefix.
run_python "${ENV_ROOT}/pi3x" -m pip install \
  torch==2.5.1 torchvision==0.20.1 --index-url "${PI3_TORCH_INDEX_URL}"
run_python "${ENV_ROOT}/pi3x" -m pip install -r "${SOURCE_ROOT}/Pi3/requirements.txt"
run_python "${ENV_ROOT}/pi3x" -m pip install -e "${SOURCE_ROOT}/Pi3"
run_python "${ENV_ROOT}/pi3x" -m pip install --no-deps -e "${PROJECT_ROOT}"
run_python "${ENV_ROOT}/pi3x" -m pip install scipy tqdm

# MoGe-3 is kept separate because it requires NumPy 2.x and Triton/FlexGEMM.
run_python "${ENV_ROOT}/moge3" -m pip install \
  torch torchvision --index-url "${MOGE_TORCH_INDEX_URL}"
run_python "${ENV_ROOT}/moge3" -m pip install -e "${SOURCE_ROOT}/MoGe"

# VIPE compiles CUDA extensions against the Torch installed in this prefix.
run_python "${ENV_ROOT}/vipe" -m pip install \
  torch torchvision --index-url "${VIPE_TORCH_INDEX_URL}"
run_python "${ENV_ROOT}/vipe" "${PROJECT_ROOT}/scripts/setup_vipe.py" \
  --vipe-source "${SOURCE_ROOT}/vipe"

mkdir -p "${PROJECT_ROOT}/ckpt/pi3x" "${PROJECT_ROOT}/ckpt/moge3" "${PROJECT_ROOT}/ckpt/vipe"
run_python "${ENV_ROOT}/pi3x" "${PROJECT_ROOT}/scripts/check_three_envs.py" \
  --env-root "${ENV_ROOT}" --project-root "${PROJECT_ROOT}" --skip-checkpoints

echo "Three Conda-prefix environments created under ${ENV_ROOT}"
echo "Run without activation: ${ENV_ROOT}/pi3x/bin/python ${PROJECT_ROOT}/cli.py --help"
