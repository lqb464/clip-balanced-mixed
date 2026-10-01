import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


setup = '''import os
import subprocess
import sys
from pathlib import Path

repo = Path("/kaggle/working/clip-balanced-mixed")
if not repo.exists():
    subprocess.run([
        "git", "clone", "https://github.com/lqb464/clip-balanced-mixed.git", str(repo)
    ], check=True)

requirements = repo / "requirements.txt"
dependencies = []
for line in requirements.read_text(encoding="utf-8").splitlines():
    package = line.strip()
    if package and not package.startswith(("#", "torch")):
        dependencies.append(package)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *dependencies], check=True)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("Hãy bật GPU trong Kaggle trước khi chạy cell huấn luyện.")

data_root = Path("/kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark")
if not (data_root / "RSTPReid" / "data_captions.json").is_file():
    raise FileNotFoundError(f"Không tìm thấy annotation RSTPReid tại {data_root}")

os.environ["TBPS_REPO"] = str(repo)
os.environ["TBPS_DATA_ROOT"] = str(data_root)
print(f"GPU đang dùng: {torch.cuda.get_device_name(0)}")
print(f"Repo: {repo}")
print(f"Dataset: {data_root / 'RSTPReid'}")
'''

launch_cells = {
    "irra": '''%%bash
set -euo pipefail
cd "$TBPS_REPO/irra"
CUDA_VISIBLE_DEVICES=0 python -u train.py \\
    --name balanced_mixed_rstpreid \\
    --img_aug \\
    --batch_size 64 \\
    --MLM \\
    --dataset_name RSTPReid \\
    --loss_names 'sdm+mlm+id' \\
    --num_epoch 60 \\
    --root_dir "$TBPS_DATA_ROOT" \\
    --sampler balanced_mixed \\
    --positive-pairs 4 \\
    --sampler-refresh-every 3 \\
    --seed 1
''',
    "dm-adapter": '''# %%bash
# set -euo pipefail
# cd "$TBPS_REPO/dm-adapter"
# CUDA_VISIBLE_DEVICES=0 python -u train.py \\
#     --name balanced_mixed_rstpreid \\
#     --img_aug \\
#     --batch_size 128 \\
#     --MLM \\
#     --dataset_name RSTPReid \\
#     --loss_names 'sdm+aux' \\
#     --num_epoch 60 \\
#     --root_dir "$TBPS_DATA_ROOT" \\
#     --lr 3e-4 \\
#     --num_experts 6 \\
#     --topk 2 \\
#     --reduction 8 \\
#     --sampler balanced_mixed \\
#     --positive-pairs 4 \\
#     --sampler-refresh-every 3 \\
#     --seed 1
''',
    "rde": '''# %%bash
# set -euo pipefail
# cd "$TBPS_REPO/rde/2024-CVPR-RDE"
# CUDA_VISIBLE_DEVICES=0 python -u train.py \\
#     --noisy_rate 0.0 \\
#     --noisy_file ./noiseindex/RSTPReid_0.0.npy \\
#     --name RDE-Balanced-Mixed \\
#     --img_aug \\
#     --txt_aug \\
#     --batch_size 64 \\
#     --select_ratio 0.3 \\
#     --tau 0.015 \\
#     --root_dir "$TBPS_DATA_ROOT" \\
#     --output_dir run_logs_balanced_mixed \\
#     --margin 0.1 \\
#     --dataset_name RSTPReid \\
#     --loss_names 'TAL+sr0.3_tau0.015_margin0.1_n0.0' \\
#     --num_epoch 60 \\
#     --lr-total-epochs 60 \\
#     --sampler balanced_mixed \\
#     --positive-pairs 4 \\
#     --sampler-refresh-every 3 \\
#     --seed 1
''',
    "itself": '''# %%bash
# set -euo pipefail
# cd "$TBPS_REPO/itself"
# CUDA_VISIBLE_DEVICES=0 python -u train.py \\
#     --name ITSELF-Balanced-Mixed \\
#     --output_dir ITSELF-balanced-mixed \\
#     --batch_size 256 \\
#     --dataset_name RSTPReid \\
#     --root_dir "$TBPS_DATA_ROOT" \\
#     --loss_names 'tal+cid' \\
#     --num_epoch 60 \\
#     --only_global \\
#     --sampler balanced_mixed \\
#     --positive-pairs 4 \\
#     --sampler-refresh-every 3 \\
#     --seed 1
''',
}

cells = [
    markdown("""# Balanced Mixed cho TBPS\n\nNotebook chạy các pipeline IRRA, DM-Adapter, RDE và ITSELF trên RSTPReid theo đúng thứ tự này. Mỗi phương pháp giữ cấu hình train riêng và dùng chung chính sách Balanced Mixed. Cell IRRA đang mở; ba cell còn lại được comment. Bỏ comment cell của phương pháp bạn muốn chạy.\n\nMỗi cell là một lượt train riêng 60 epochs, có thể vượt thời lượng một Kaggle session. Mining dùng trọng số model hiện tại và cập nhật mỗi ba epochs.\n"""),
    code(setup),
    markdown("""## 1. IRRA\n\nBatch size 64; `sdm+mlm+id`; bật image augmentation và MLM.\n"""),
    code(launch_cells["irra"]),
    markdown("""## 2. DM-Adapter\n\nBatch size 128; `sdm+aux`; bật image augmentation và MLM.\n"""),
    code(launch_cells["dm-adapter"]),
    markdown("""## 3. RDE\n\nBatch size 64; cấu hình TAL/RBS từ launcher RDE trong repo; noise rate 0.\n"""),
    code(launch_cells["rde"]),
    markdown("""## 4. ITSELF\n\nBatch size 256; `tal+cid`; bật `only_global`.\n"""),
    code(launch_cells["itself"]),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(ROOT / "kaggle_balanced_mixed.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
