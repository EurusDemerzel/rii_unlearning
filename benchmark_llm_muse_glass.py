#!/usr/bin/env python3
"""
benchmark_llm_muse_glass.py — MUSE-style long-form book unlearning, retain
variant: Through the Looking-Glass (same author as Alice -> comparable
forget/retain output distributions, lower RII baseline).

Reuses the teacher / NegGrad artifacts from benchmark_llm_muse.py:
  teacher adapter: results/benchmark_llm_muse/adapter_teacher.safetensors
  neggrad adapter: results/benchmark_llm_muse/adapter_neggrad.safetensors
Only FineTune and Retrain are re-run on the Looking-Glass retain set.

Metrics (same as benchmark_llm_muse.py):
  forget_ppl / retain_ppl, RII (2 x V), MIA (= log forget_ppl), forget_rouge
Outputs: results/benchmark_llm_muse/results_glass.csv
"""
import os, time, csv, math, pickle
import numpy as np

import benchmark_llm_muse as m

OUT = os.path.join("results", "benchmark_llm_muse")
DATA_PKL = os.path.join("data", "muse", "muse_data.pkl")
DATA_GLASS = os.path.join("data", "muse", "muse_data_glass.pkl")
TEACHER_ADAPTER = os.path.join(OUT, "adapter_teacher.safetensors")
NEGGRAD_ADAPTER = os.path.join(OUT, "adapter_neggrad.safetensors")


def load_pickle(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def evaluate(tag, model, tokenizer, ev_forget, ev_retain, do_rouge=True):
    f_nll = m.mean_nll(model, tokenizer, ev_forget)
    r_nll = m.mean_nll(model, tokenizer, ev_retain)
    rho = m.rii_llm(model, tokenizer, ev_forget, ev_retain)
    f_rouge = m.continuation_rouge(model, tokenizer, ev_forget) if do_rouge else float("nan")
    res = dict(method=tag,
               forget_ppl=math.exp(f_nll), retain_ppl=math.exp(r_nll),
               rii=rho, mia=f_nll, forget_rouge=f_rouge)
    print(f"  [{tag:10s}] forget_ppl={res['forget_ppl']:7.2f} retain_ppl={res['retain_ppl']:7.2f} "
          f"RII={rho:.4f} MIA={f_nll:.3f} forget_rouge={f_rouge:.1f}%")
    return res


def main():
    t0 = time.time()
    data = load_pickle(DATA_PKL)
    glass = load_pickle(DATA_GLASS)
    ev_forget = data["ev_forget"]
    ev_retain = glass["ev_retain"]
    tr_retain = glass["tr_retain"]
    print(f"data: ev_forget={len(ev_forget)} ev_retain={len(ev_retain)} tr_retain={len(tr_retain)}")

    rows = []

    # NoUnlearn: teacher (base + Alice LoRA)
    print("\n[1] NoUnlearn (teacher) ...")
    model, tokenizer = m.load_model(m.BASE_MODEL_DIR)
    m.load_teacher_lora(model, TEACHER_ADAPTER)
    rows.append(evaluate("NoUnlearn", model, tokenizer, ev_forget, ev_retain))

    # FineTune: teacher LoRA continue on Looking-Glass
    print(f"\n[2] FineTune: teacher LoRA continue on Looking-Glass, {m.FT_STEPS} steps ...")
    m.finetune_lora(model, tokenizer, tr_retain,
                    os.path.join(OUT, "adapter_finetune_glass.safetensors"))
    rows.append(evaluate("FineTune", model, tokenizer, ev_forget, ev_retain))

    # NegGrad: reuse 15-step adapter from main run
    print("\n[3] NegGrad (reuse 15-step adapter) ...")
    model, tokenizer = m.load_model(m.BASE_MODEL_DIR)
    m.load_teacher_lora(model, NEGGRAD_ADAPTER)
    rows.append(evaluate("NegGrad", model, tokenizer, ev_forget, ev_retain))

    # Retrain oracle: base + LoRA on Looking-Glass only
    print(f"\n[4] Retrain oracle: base + LoRA on Looking-Glass only, {m.FT_STEPS} steps ...")
    model, tokenizer = m.load_model(m.BASE_MODEL_DIR)
    m.add_lora(model)
    m.finetune_lora(model, tokenizer, tr_retain,
                    os.path.join(OUT, "adapter_retrain_glass.safetensors"))
    rows.append(evaluate("Retrain", model, tokenizer, ev_forget, ev_retain))

    with open(os.path.join(OUT, "results_glass.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "forget_ppl", "retain_ppl", "rii", "mia", "forget_rouge"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nsaved:", os.path.join(OUT, "results_glass.csv"))
    print("=== summary (retain = Looking-Glass) ===")
    for r in rows:
        print(f"  {r['method']:10s} forget_ppl={r['forget_ppl']:7.2f} retain_ppl={r['retain_ppl']:7.2f} "
              f"RII={r['rii']:.4f} MIA={r['mia']:.3f} rouge={r['forget_rouge']:.1f}%")
    print(f"\nall done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
