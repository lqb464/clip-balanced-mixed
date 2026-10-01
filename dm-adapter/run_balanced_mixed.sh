#!/bin/bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-RSTPReid}"
ROOT_DIR="${TBPS_DATA_ROOT:-/kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="$GPU" python -u train.py \
    --name balanced_mixed_rstpreid \
    --img_aug \
    --batch_size 128 \
    --MLM \
    --dataset_name "$DATASET_NAME" \
    --loss_names 'sdm+aux' \
    --num_epoch 60 \
    --root_dir "$ROOT_DIR" \
    --lr 3e-4 \
    --num_experts 6 \
    --topk 2 \
    --reduction 8 \
    --sampler balanced_mixed \
    --positive-pairs 4 \
    --sampler-refresh-every 3 \
    --seed 1
