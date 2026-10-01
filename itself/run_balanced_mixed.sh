#!/bin/bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-RSTPReid}"
ROOT_DIR="${TBPS_DATA_ROOT:-/kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="$GPU" python -u train.py \
    --name ITSELF-Balanced-Mixed \
    --output_dir ITSELF-balanced-mixed \
    --batch_size 256 \
    --dataset_name "$DATASET_NAME" \
    --root_dir "$ROOT_DIR" \
    --loss_names 'tal+cid' \
    --num_epoch 60 \
    --only_global \
    --sampler balanced_mixed \
    --positive-pairs 4 \
    --sampler-refresh-every 3 \
    --seed 1
