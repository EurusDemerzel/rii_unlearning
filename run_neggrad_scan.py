#!/usr/bin/env python3
"""
run_neggrad_scan.py — NegGrad strength sweep on TOFU LLaMA-2-7B (MLX+LoRA).

Responds to the reviewer concern about gradient-ascent strength: we sweep the
ascent strength (steps x lr) 3-5x beyond the main-run setting and re-measure
forget/retain accuracy, RII and MIA. This shows RII's *bidirectional*
sensitivity:
  - if the model collapses (acc -> 0, output corruption), RII should -> ~0
    (channel degeneracy) while MIA explodes -> explicit collapse detection;
  - if accuracy stays high but RII keeps rising, the hidden-failure signal
    persists (RII is not just an accuracy artifact).

Each strength starts from a FRESH zero LoRA on tofu_ft (== original model), so
the sweep is an ablation of strength only. Output:
results/benchmark_llm_tofu_mlx/results_neggrad_scan.csv
"""
import os, json, time, csv, random
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from benchmark_llm_tofu_mlx import (load_tofu, load_model, add_lora, evaluate,
                                    neggrad_ascent, make_text, OUT,
                                    FT_MODEL_DIR)

# strength grid: (steps, lr) — main run is (10, 2e-5)
GRID = [
    ("weak_base", 10, 2e-5),     # baseline NegGrad (already in results.csv)
    ("mid_x3",    30, 1e-4),     # 3x steps, 5x lr
    ("strong_x6", 60, 2e-4),     # 6x steps, 10x lr
]


def main():
    t0 = time.time()
    ev_forget, ev_retain, _ = load_tofu()
    forget_texts = [make_text(q, a) for q, a in ev_forget]
    rows = []
    for tag, steps, lr in GRID:
        print(f"\n=== NegGrad strength [{tag}] steps={steps} lr={lr:.0e} ===")
        model, tokenizer = load_model(FT_MODEL_DIR)
        add_lora(model)          # fresh zero LoRA == original model
        adapter = os.path.join(OUT, f"adapter_neggrad_{tag}.safetensors")
        neggrad_ascent(model, tokenizer, forget_texts, adapter, steps=steps, lr=lr)
        rows.append(evaluate(tag, model, tokenizer, ev_forget, ev_retain))

    with open(os.path.join(OUT, "results_neggrad_scan.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "forget_acc", "retain_acc",
                                           "rii", "mia"])
        w.writeheader()
        w.writerows(rows)
    print("\nsaved:", os.path.join(OUT, "results_neggrad_scan.csv"))
    print("=== NegGrad strength sweep summary ===")
    for r in rows:
        print(f"  {r['method']:12s} forget={r['forget_acc']:5.1f}% retain={r['retain_acc']:5.1f}% "
              f"RII={r['rii']:.4f} MIA={r['mia']:.3f}")
    print(f"\nall done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
