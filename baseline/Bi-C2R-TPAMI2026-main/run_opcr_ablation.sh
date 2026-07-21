#!/usr/bin/env bash
set -euo pipefail

EXP="${1:-all}"
GPU="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
SETTING="${3:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "${SCRIPT_DIR}" == "${SCRIPT_PATH}" ]]; then
  SCRIPT_DIR="."
fi
SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"

cd "${SCRIPT_DIR}"

SUITE=(
  grn_fixed_005
  grn_opcr_rel03_cycle005
  grn_opcr_rel05_cycle01
  grn_opcr_rel07_cycle01
  grn_opcr_rel05_no_cycle
  opcr_no_grn_rel05_cycle01
)

usage() {
  echo "Usage: bash run_opcr_ablation.sh [experiment|all] [gpu_id] [setting: 1|2]"
  echo "Experiments:"
  echo "  all"
  echo "  baseline"
  printf '  %s\n' "${SUITE[@]}"
}

if [[ "${SETTING}" != "1" && "${SETTING}" != "2" ]]; then
  usage
  exit 1
fi

run_one() {
  local name="$1"
  local cfg="config/${name}.yml"
  local log_dir="logs-opcr-setting${SETTING}/${name}/"

  if [[ "${name}" == "baseline" ]]; then
    cfg="config/base.yml"
    log_dir="logs-opcr-setting${SETTING}/baseline/"
  fi

  if [[ ! -f "${cfg}" ]]; then
    echo "Missing config: ${cfg}"
    exit 1
  fi

  echo "Running ${name} on setting=${SETTING}, GPU=${GPU}, config=${cfg}, logs=${log_dir}"
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN=1, skip training."
    return
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" continual_train.py \
    --config_file "${cfg}" \
    --logs-dir "${log_dir}" \
    --setting "${SETTING}"
}

if [[ "${EXP}" == "all" ]]; then
  for name in "${SUITE[@]}"; do
    run_one "${name}"
  done
else
  run_one "${EXP}"
fi
