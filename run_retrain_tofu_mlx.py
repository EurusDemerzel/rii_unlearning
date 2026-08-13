#!/usr/bin/env python3
"""
run_retrain_tofu_mlx.py — Retrain oracle only (base LLaMA-2-7B + LoRA on retain).
Reuses benchmark_llm_tofu_mlx functions; appends Retrain row to results.csv.

NOTE: uses a custom no-padding LoRA loop. Official mlx_lm train() pads with
token 0 (=<unk>), which yields NaN on the un-finetuned base LLaMA-2-7B.
"""
import os, csv, time, random
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.tuner.trainer import default_loss
from mlx.utils import tree_flatten
from benchmark_llm_tofu_mlx import (load_tofu, load_model, add_lora,
                                    evaluate, make_text,
                                    OUT, BASE_MODEL_DIR, MAX_SEQ)


def finetune_custom(model, tokenizer, texts, adapter_file, steps=60, lr=1e-6):
    """LoRA fine-tune on raw (no-pad) single-sample batches (base-safe)."""
    def lf(m, b, l):
        return default_loss(m, b, l)[0]
    lag = nn.value_and_grad(model, lf)
    opt = optim.AdamW(learning_rate=lr)
    for it in range(steps):
        t = random.choice(texts)
        ids = tokenizer.encode(t)[:MAX_SEQ]
        if len(ids) < 3:
            continue
        b = mx.array([ids])
        l = mx.array([[0, max(len(ids) - 2, 1)]])
        lv, gr = lag(model, b, l)
        opt.update(model, gr)
        if (it + 1) % 10 == 0 or it == steps - 1:
            print(f"      step {it+1}/{steps} loss={float(lv):.4f}")
    mx.save_safetensors(adapter_file, dict(tree_flatten(model.trainable_parameters())))
    return adapter_file


def main():
    t0 = time.time()
    ev_forget, ev_retain, tr_retain = load_tofu()
    assert os.path.isdir(BASE_MODEL_DIR), f"missing {BASE_MODEL_DIR}"
    print(f"[Retrain] base {os.path.basename(BASE_MODEL_DIR)} + LoRA retain only")
    model, tokenizer = load_model(BASE_MODEL_DIR)
    add_lora(model)
    finetune_custom(model, tokenizer,
                    [make_text(q, a) for q, a in tr_retain],
                    os.path.join(OUT, "adapter_retrain.safetensors"))
    row = evaluate("Retrain", model, tokenizer, ev_forget, ev_retain)
    # merge into results.csv
    csv_path = os.path.join(OUT, "results.csv")
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
    rows = [r for r in rows if r["method"] != "Retrain"]
    rows.append({k: (f"{v:.4f}" if isinstance(v, float) else v)
                 for k, v in row.items()})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "forget_acc", "retain_acc", "rii", "mia"])
        w.writeheader()
        w.writerows(rows)
    print("\nsaved:", csv_path)
    for r in rows:
        print(f"  {r['method']:12s} forget={r['forget_acc']:>6}% retain={r['retain_acc']:>6}% "
              f"RII={r['rii']} MIA={r['mia']}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
