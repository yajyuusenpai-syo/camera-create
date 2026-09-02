#!/usr/bin/env bash
# Clone the three upstream model repositories without adding them to camera-create Git.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/third_party}"
PI3_REF="${PI3_REF:-main}"
MOGE_REF="${MOGE_REF:-main}"
VIPE_REF="${VIPE_REF:-v1.2.0}"

clone_or_update() {
  local repository="$1"
  local destination="$2"
  local revision="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone "${repository}" "${destination}"
  fi
  git -C "${destination}" fetch --tags origin
  git -C "${destination}" checkout --detach "${revision}"
}

mkdir -p "${SOURCE_ROOT}"
clone_or_update "https://github.com/yyfz/Pi3.git" "${SOURCE_ROOT}/Pi3" "${PI3_REF}"
clone_or_update "https://github.com/microsoft/MoGe.git" "${SOURCE_ROOT}/MoGe" "${MOGE_REF}"
clone_or_update "https://github.com/nv-tlabs/vipe.git" "${SOURCE_ROOT}/vipe" "${VIPE_REF}"

{
  echo "Pi3 $(git -C "${SOURCE_ROOT}/Pi3" rev-parse HEAD)"
  echo "MoGe $(git -C "${SOURCE_ROOT}/MoGe" rev-parse HEAD)"
  echo "VIPE $(git -C "${SOURCE_ROOT}/vipe" rev-parse HEAD)"
} > "${SOURCE_ROOT}/SOURCE_VERSIONS.txt"

echo "Upstream sources ready in ${SOURCE_ROOT}"
echo "Resolved revisions written to ${SOURCE_ROOT}/SOURCE_VERSIONS.txt"
