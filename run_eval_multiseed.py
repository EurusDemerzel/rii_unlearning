#!/usr/bin/env python3
"""
run_eval_multiseed.py — multi-seed (3×) evaluation of RII/MIA on the TOFU LLM
experiment, using subsampled eval sets per seed (no retraining).

Responds to the reviewer concern about single-seed / missing error bars:
reports mean ± std of RII and MIA over 3 random subsamples of the evaluation
set for each of the 4 methods. Accuracy is inherited from the main run.
Outputs: results/benchmark_llm_tofu_mlx/results_multiseed.csv
"""
import os, json, csv, random
import numpy as np
import mlx.core as mx
from mlx_lm import load

from benchmark_llm_tofu_mlx import (load_tofu, add_lora, rii_llm, mean_nll,
                                    make_text, OUT, FT_MODEL_DIR, BASE_MODEL_DIR)

SEEDS = [0, 1, 2]
SUB_F = 15   # subsample from 20 forget QA
SUB_R = 30   # subsample from 40 retain QA


def build_model(kind):
    if kind == "NoUnlearn":
        model, tok = load(FT_MODEL_DIR)
    elif kind == "Retrain":
        model, tok = load(BASE_MODEL_DIR)
        add_lora(model)
        model.load_weights(mx.load(os.path.join(OUT, "adapter_retrain.safetensors")),
                           strict=False)
    else:
        model, tok = load(FT_MODEL_DIR)
        add_lora(model)
        model.load_weights(
            mx.load(os.path.join(OUT, f"adapter_{kind.lower()}.safetensors")),
            strict=False)
    return model, tok


def main():
    ev_forget, ev_retain, _ = load_tofu()
    print("methods × seeds evaluation (subsampled eval set per seed) ...")
    rows = []
    for kind in ["NoUnlearn", "FineTune", "NegGrad", "Retrain"]:
        model, tok = build_model(kind)
        rhos, mias = [], []
        for seed in SEEDS:
            random.seed(seed)
            sub_f = random.sample(ev_forget, SUB_F)
            sub_r = random.sample(ev_retain, SUB_R)
            rho = rii_llm(model, tok, sub_f, sub_r)
            mia = mean_nll(model, tok, sub_f)
            rhos.append(rho); mias.append(mia)
            print(f"  {kind:10s} seed={seed}: RII={rho:.4f} MIA={mia:.3f}")
        rho_m, rho_s = np.mean(rhos), np.std(rhos)
        mia_m, mia_s = np.mean(mias), np.std(mias)
        print(f"  {kind:10s} => RII={rho_m:.4f}±{rho_s:.4f}  MIA={mia_m:.3f}±{mia_s:.3f}")
        rows.append(dict(method=kind, rii_mean=rho_m, rii_std=rho_s,
                         mia_mean=mia_m, mia_std=mia_s))

    with open(os.path.join(OUT, "results_multiseed.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "rii_mean", "rii_std",
                                           "mia_mean", "mia_std"])
        w.writeheader()
        w.writerows(rows)
    print("\nsaved:", os.path.join(OUT, "results_multiseed.csv"))


if __name__ == "__main__":
    main()
