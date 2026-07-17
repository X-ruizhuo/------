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
  grn_fixed_005_plus
  grn_fixed_007
  grn_fixed_007_plus
  grn_fixed_010_stable
  grn_fixed_005_strong_old
  grn_fixed_007_discri
)

usage() {
  echo "Usage: bash run_grn_next_ablation.sh [experiment|all] [gpu_id] [setting: 1|2]"
  echo "Experiments:"
  echo "  all"
  echo "  baseline"
  echo "  grn_fixed_005"
  printf '  %s\n' "${SUITE[@]}"
}

if [[ "${SETTING}" != "1" && "${SETTING}" != "2" ]]; then
  usage
  exit 1
fi

run_one() {
  local name="$1"
  local cfg="config/${name}.yml"
  local log_dir="logs-grn-next-setting${SETTING}/${name}/"
  local args=()

  case "${name}" in
    baseline)
      cfg="config/base.yml"
      log_dir="logs-grn-next-setting${SETTING}/baseline/"
      ;;
    grn_fixed_005)
      cfg="config/grn_fixed_005.yml"
      ;;
    grn_fixed_005_plus)
      args+=(--AF_weight 1.2 --weight_trans 120)
      ;;
    grn_fixed_007)
      ;;
    grn_fixed_007_plus)
      args+=(--AF_weight 1.2 --weight_trans 120)
      ;;
    grn_fixed_010_stable)
      args+=(--lr 0.006 --warmup-step 15 --weight_trans 120)
      ;;
    grn_fixed_005_strong_old)
      args+=(--lr 0.006 --AF_weight 1.5 --weight_trans 150)
      ;;
    grn_fixed_007_discri)
      args+=(--weight_discri 0.010 --weight_transx 0.001)
      ;;
    *)
      usage
      exit 1
      ;;
  esac

  if [[ ! -f "${cfg}" ]]; then
    echo "Missing config: ${cfg}"
    exit 1
  fi

  echo "Running ${name} on setting=${SETTING}, GPU=${GPU}, config=${cfg}, logs=${log_dir}"
  if [[ ${#args[@]} -gt 0 ]]; then
    echo "Extra args: ${args[*]}"
  else
    echo "Extra args: <baseline training args>"
  fi
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN=1, skip training."
    return
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" continual_train.py \
    --config_file "${cfg}" \
    --logs-dir "${log_dir}" \
    --setting "${SETTING}" \
    "${args[@]}"
}

if [[ "${EXP}" == "all" ]]; then
  for name in "${SUITE[@]}"; do
    run_one "${name}"
  done
else
  run_one "${EXP}"
fi
