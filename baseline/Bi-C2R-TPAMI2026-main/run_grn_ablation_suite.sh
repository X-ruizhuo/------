#!/usr/bin/env bash
set -euo pipefail

SETTING="${1:-1}"
GPU="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${SETTING}" != "1" && "${SETTING}" != "2" ]]; then
  echo "Usage: bash run_grn_ablation_suite.sh [setting: 1|2] [gpu_id]"
  exit 1
fi

SUITE=(
  grn_noop
  grn_safe_005
  grn_safe_010
  grn_fixed_005
  grn_no_beta
  grn_detach_response
)

run_one() {
  local name="$1"
  local cfg="config/${name}.yml"
  local log_dir="logs-grn-setting${SETTING}/${name}/"

  echo "Running ${name} on setting=${SETTING}, GPU=${GPU}, config=${cfg}, logs=${log_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" continual_train.py \
    --config_file "${cfg}" \
    --logs-dir "${log_dir}" \
    --setting "${SETTING}"
}

for name in "${SUITE[@]}"; do
  run_one "${name}"
done
