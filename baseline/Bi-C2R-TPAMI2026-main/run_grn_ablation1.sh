#!/usr/bin/env bash
set -euo pipefail

EXP="${1:-all}"
GPU="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
SETTING=1
SUITE=(baseline grn_noop grn_safe_005 grn_safe_010 grn_fixed_005 grn_no_beta grn_detach_response)

run_one() {
  local name="$1"
  local cfg="config/${name}.yml"
  if [[ "${name}" == "baseline" ]]; then
    cfg="config/base.yml"
  fi
  echo "Running ${name} on setting=${SETTING}, GPU=${GPU}, config=${cfg}"
  CUDA_VISIBLE_DEVICES="${GPU}" python continual_train.py \
    --config_file "${cfg}" \
    --logs-dir "logs-grn-setting${SETTING}/${name}/" \
    --setting "${SETTING}"
}

if [[ "${EXP}" == "all" ]]; then
  for name in "${SUITE[@]}"; do
    run_one "${name}"
  done
else
  run_one "${EXP}"
fi
