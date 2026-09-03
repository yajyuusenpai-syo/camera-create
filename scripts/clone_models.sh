#!/usr/bin/env bash
# Clone the three upstream model repositories without adding them to camera-create Git.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/third_party}"
PI3_REF="${PI3_REF:-main}"
MOGE_REF="${MOGE_REF:-main}"
VIPE_REF="${VIPE_REF:-v1.2.0}"
UTILS3D_MOGE_REF="${UTILS3D_MOGE_REF:-62f09d58509485564e24d5d9f6aac9ee9ebc0c37}"
PIPELINE_REF="${PIPELINE_REF:-1c511390d90226c00c101f34b84df26a0f8789b4}"
FLEX_GEMM_REF="${FLEX_GEMM_REF:-b2fadb29d41846c7981ade6801ffc689fae119cf}"
GIT_RETRIES="${GIT_RETRIES:-3}"

retry_git() {
  local attempt
  for ((attempt = 1; attempt <= GIT_RETRIES; attempt++)); do
    if git "$@"; then
      return 0
    fi
    echo "Git command failed (${attempt}/${GIT_RETRIES}); retrying in 5 seconds..." >&2
    sleep 5
  done
  return 1
}

clone_or_update() {
  local repository="$1"
  local destination="$2"
  local revision="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    retry_git clone --filter=blob:none "${repository}" "${destination}"
  elif git -C "${destination}" checkout --detach "${revision}"; then
    return 0
  fi
  retry_git -C "${destination}" fetch --tags origin
  git -C "${destination}" checkout --detach "${revision}"
}

mkdir -p "${SOURCE_ROOT}"
clone_or_update "https://github.com/yyfz/Pi3.git" "${SOURCE_ROOT}/Pi3" "${PI3_REF}"
clone_or_update "https://github.com/microsoft/MoGe.git" "${SOURCE_ROOT}/MoGe" "${MOGE_REF}"
clone_or_update "https://github.com/nv-tlabs/vipe.git" "${SOURCE_ROOT}/vipe" "${VIPE_REF}"
clone_or_update "https://github.com/EasternJournalist/utils3d-moge.git" \
  "${SOURCE_ROOT}/utils3d-moge" "${UTILS3D_MOGE_REF}"
clone_or_update "https://github.com/EasternJournalist/pipeline.git" \
  "${SOURCE_ROOT}/pipeline" "${PIPELINE_REF}"
clone_or_update "https://github.com/JeffreyXiang/FlexGEMM.git" \
  "${SOURCE_ROOT}/FlexGEMM" "${FLEX_GEMM_REF}"

{
  echo "Pi3 $(git -C "${SOURCE_ROOT}/Pi3" rev-parse HEAD)"
  echo "MoGe $(git -C "${SOURCE_ROOT}/MoGe" rev-parse HEAD)"
  echo "VIPE $(git -C "${SOURCE_ROOT}/vipe" rev-parse HEAD)"
  echo "utils3d-moge $(git -C "${SOURCE_ROOT}/utils3d-moge" rev-parse HEAD)"
  echo "pipeline $(git -C "${SOURCE_ROOT}/pipeline" rev-parse HEAD)"
  echo "FlexGEMM $(git -C "${SOURCE_ROOT}/FlexGEMM" rev-parse HEAD)"
} > "${SOURCE_ROOT}/SOURCE_VERSIONS.txt"

echo "Upstream sources ready in ${SOURCE_ROOT}"
echo "Resolved revisions written to ${SOURCE_ROOT}/SOURCE_VERSIONS.txt"
