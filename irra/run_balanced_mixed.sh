#!/bin/bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-RSTPReid}"
ROOT_DIR="${TBPS_DATA_ROOT:-/kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="$GPU" python -u train.py \
    --name balanced_mixed_rstpreid \
    --img_aug \
    --batch_size 64 \
    --MLM \
    --dataset_name "$DATASET_NAME" \
    --loss_names 'sdm+mlm+id' \
    --num_epoch 60 \
    --root_dir "$ROOT_DIR" \
    --sampler balanced_mixed \
    --positive-pairs 4 \
    --sampler-refresh-every 3 \
    --seed 1
