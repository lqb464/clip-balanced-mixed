#!/bin/bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-RSTPReid}"
ROOT_DIR="${TBPS_DATA_ROOT:-/kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
NOISY_RATE=0.0
NOISY_FILE="./noiseindex/${DATASET_NAME}_${NOISY_RATE}.npy"
TAU=0.015
MARGIN=0.1
SELECT_RATIO=0.3

CUDA_VISIBLE_DEVICES="$GPU" python -u train.py \
    --noisy_rate "$NOISY_RATE" \
    --noisy_file "$NOISY_FILE" \
    --name RDE-Balanced-Mixed \
    --img_aug \
    --txt_aug \
    --batch_size 64 \
    --select_ratio "$SELECT_RATIO" \
    --tau "$TAU" \
    --root_dir "$ROOT_DIR" \
    --output_dir run_logs_balanced_mixed \
    --margin "$MARGIN" \
    --dataset_name "$DATASET_NAME" \
    --loss_names "TAL+sr${SELECT_RATIO}_tau${TAU}_margin${MARGIN}_n${NOISY_RATE}" \
    --num_epoch 60 \
    --lr-total-epochs 60 \
    --sampler balanced_mixed \
    --positive-pairs 4 \
    --sampler-refresh-every 3 \
    --seed 1
