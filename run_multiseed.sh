#!/usr/bin/env bash
# run_multiseed.sh — run the class-level visual benchmarks over seeds 0,1,2.
# Usage: bash run_multiseed.sh [logdir]
set -euo pipefail
PY=./.venv/bin/python
LOG="${1:-/tmp/multiseed}"
mkdir -p "$LOG"

for s in 0 1 2; do
  echo "=== CIFAR-10 seed $s ==="
  $PY benchmark_v2.py --seed "$s" > "$LOG/cifar10_s$s.log" 2>&1
  echo "  done (see $LOG/cifar10_s$s.log)"
done

for s in 0 1 2; do
  echo "=== Fashion-MNIST seed $s ==="
  $PY benchmark_v2.py --fashion_mnist --seed "$s" > "$LOG/fashion_s$s.log" 2>&1
  echo "  done (see $LOG/fashion_s$s.log)"
done

for s in 0 1 2; do
  echo "=== CIFAR-100 seed $s ==="
  $PY benchmark_cifar100.py --seed "$s" > "$LOG/cifar100_s$s.log" 2>&1
  echo "  done (see $LOG/cifar100_s$s.log)"
done

echo "ALL MULTISEED RUNS COMPLETE"
