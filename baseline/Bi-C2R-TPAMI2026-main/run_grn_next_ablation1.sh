#!/usr/bin/env bash
set -euo pipefail

EXP="${1:-all}"
GPU="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "${SCRIPT_DIR}" == "${SCRIPT_PATH}" ]]; then
  SCRIPT_DIR="."
fi
SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"

"${BASH:-bash}" "${SCRIPT_DIR}/run_grn_next_ablation.sh" "${EXP}" "${GPU}" 1
