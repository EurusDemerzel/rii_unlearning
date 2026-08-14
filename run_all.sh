#!/usr/bin/env bash
# run_all.sh — reproduce the main visual benchmark tables of the paper.
#
# Requires:
#   - the virtual environment (run `bash setup.sh` first)
#   - the datasets under ./data (auto-downloaded by torchvision on first run)
#
# Usage:  bash run_all.sh
set -euo pipefail

if [ ! -d "venv" ]; then
  echo "[ERROR] venv not found. Run: bash setup.sh"
  exit 1
fi
source venv/bin/activate

echo "[1/5] CIFAR-10 class-level benchmark (Table tab:benchmark) ..."
python benchmark_v2.py --seed 0

echo "[2/5] Fashion-MNIST benchmark (Table tab:cross) ..."
python benchmark_v2.py --fashion_mnist --seed 0

echo "[3/5] CIFAR-100 benchmark (Table tab:cifar100) ..."
python benchmark_cifar100.py --seed 0

echo "[4/5] MNIST MHPR / K-ablation (Table tab:mhpr, Fig. k_ablation) ..."
python run_multi_heldout_projection.py

echo "[5/5] Done. Results are written under results/"
