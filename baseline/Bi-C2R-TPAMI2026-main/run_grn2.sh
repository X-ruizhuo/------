#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
CUDA_VISIBLE_DEVICES="${GPU}" python continual_train.py \
  --config_file config/grn.yml \
  --logs-dir logs-grn-setting2/grn/ \
  --setting 2
