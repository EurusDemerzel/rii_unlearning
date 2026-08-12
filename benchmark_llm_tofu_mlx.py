#!/usr/bin/env python3
"""
benchmark_llm_tofu_mlx.py — TOFU LLM unlearning via MLX + LoRA on Apple Silicon.

Path per user plan: quantized/quantizable LLM + PEFT (LoRA) — no full fine-tuning.
Model: Qwen2.5-1.5B (modelscope mirror). LLaMA-2-7B and the TOFU/MUSE fine-tuned
mirrors (tofu_ft_llama2-7b, MUSE-News_Llama-2-7b) are NOT available on modelscope
and hf-mirror/github-raw are unreachable, so Qwen2.5 is the viable local LLM.

Protocol (author-level TOFU):
  base Qwen2.5-1.5B
  LoRA FT on retain+forget authors  -> NoUnlearn (full-knowledge model)
  continue LoRA on retain only      -> FineTune
  LoRA gradient ASCENT on forget    -> NegGrad
  fresh LoRA on retain only (base)  -> Retrain oracle (never saw forget authors)
Metrics: forget/retain answer accuracy (ROUGE-L), RII (2 x V channel matrix on
next-token logits), MIA (mean next-token NLL).
Outputs: results/benchmark_llm_tofu_mlx/results.csv
"""
import os, json, time, csv, random, math
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)

OUT = os.path.join("results", "benchmark_llm_tofu_mlx")
os.makedirs(OUT, exist_ok=True)
DATA_DIR = os.path.join("data", "tofu")
MODEL_DIR = os.path.expanduser(
    "~/.cache/modelscope/models/Qwen--Qwen2.5-1.5B/snapshots/master")

MAX_SEQ = 384
LR = 1e-5
FT_STEPS = 60
FT_BATCH = 2
NEG_STEPS = 10
EVAL_FORGET = 20
EVAL_RETAIN = 40
GEN_MAX = 32
ROUGE_THR = 0.25


def read_jsonl(p):
    rows = []
    for line in open(p):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def make_text(q, a):
    return f"Question: {q}\n\nAnswer: {a}"


def load_tofu():
    forget = read_jsonl(os.path.join(DATA_DIR, "forget01.json"))
    retain = read_jsonl(os.path.join(DATA_DIR, "retain90.json"))
    random.shuffle(forget); random.shuffle(retain)
    forget_pairs = [(d["question"], d["answer"]) for d in forget]
    retain_pairs = [(d["question"], d["answer"]) for d in retain]
    ev_retain = retain_pairs[:EVAL_RETAIN]
    tr_retain = retain_pairs[EVAL_RETAIN:EVAL_RETAIN + 360]
    ev_forget = forget_pairs[:EVAL_FORGET]
    # full model learns retain + a slice of forget authors
    tr_full = tr_retain + forget_pairs[:30]
    return ev_forget, ev_retain, tr_retain, tr_full


def load_model():
    import mlx_lm
    model, tokenizer = mlx_lm.load(MODEL_DIR)
    return model, tokenizer


def add_lora(model, num_layers=8, rank=8, scale=20.0):
    from mlx_lm.tuner import linear_to_lora
    linear_to_lora(model, num_layers=num_layers, lora_rank=rank, lora_scale=scale)


def finetune_lora(model, texts, adapter_file, steps=FT_STEPS, lr=LR, batch=FT_BATCH):
    """LoRA fine-tune on `texts`, saving adapter to adapter_file."""
    from mlx_lm.lora import train, TrainingArgs, CacheDataset
    from mlx_lm.tuner.trainer import default_loss
    import mlx.optimizers as optim
    import mlx.core as mx
    opt = optim.AdamW(learning_rate=lr)
    args = TrainingArgs(
        batch_size=batch, iters=steps, max_seq_length=MAX_SEQ,
        steps_per_report=steps, steps_per_eval=steps + 1,
        steps_per_save=steps, adapter_file=adapter_file,
    )
    train(model=model, optimizer=opt, train_dataset=CacheDataset(texts),
          args=args, loss=default_loss)
    return adapter_file


def neggrad_ascent(model, tokenizer, texts, adapter_file, steps=NEG_STEPS, lr=LR * 2):
    """Gradient ascent on LoRA params over forget texts (NegGrad)."""
    import mlx.core as mx
    from mlx_lm.tuner.trainer import default_loss

    trainable = model.trainable_parameters()
    for _ in range(steps):
        text = random.choice(texts)
        ids = tokenizer.encode(text)
        ids = ids[:MAX_SEQ]
        toks = mx.array([ids])
        loss, grads = mx.value_and_grad(default_loss)(model, toks)
        # ascent: p <- p + lr * grad  (sign of grads is dL/dp)
        updates = mx.tree_map(lambda g: lr * g, grads)
        for k, p in trainable.items():
            p.update(p + updates[k])
        if (_ + 1) % 5 == 0 or _ == steps - 1:
            print(f"      neggrad step {_+1}/{steps} loss={float(loss):.3f}")
    mx.save_safetensors(adapter_file, dict(trainable))
    return adapter_file


def rouge_l_f1(gen, ref, tokenizer):
    g = tokenizer.encode(gen.lower())
    r = tokenizer.encode(ref.lower())
    m, n = len(g), len(r)
    if m == 0 or n == 0:
        return 0.0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        gi = g[i - 1]
        for j in range(1, n + 1):
            cur[j] = prev[j - 1] + 1 if gi == r[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[n]
    prec, rec = lcs / m, lcs / n
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def match_rate(model, tokenizer, pairs, thr=ROUGE_THR):
    from mlx_lm import generate
    hits = 0
    for q, a in pairs:
        g = generate(model, tokenizer, prompt=f"Question: {q}\n\nAnswer:",
                     max_tokens=GEN_MAX)
        if rouge_l_f1(g, a, tokenizer) >= thr:
            hits += 1
    return 100.0 * hits / max(len(pairs), 1)


def mean_nll(model, tokenizer, pairs):
    import mlx.core as mx
    from mlx_lm.tuner.trainer import default_loss
    losses = []
    for q, a in pairs:
        ids = tokenizer.encode(make_text(q, a))[:MAX_SEQ]
        losses.append(float(default_loss(model, mx.array([ids]))))
    return float(np.mean(losses))


def rii_llm(model, tokenizer, forget_pairs, retain_pairs):
    """RII from 2 x V mean next-token output distributions (V = vocab)."""
    import mlx.core as mx
    V = model.config["vocab_size"] if isinstance(model.config, dict) else model.config.vocab_size

    def mean_dist(pairs):
        acc = np.zeros(V, dtype=np.float64)
        n = 0
        for q, a in pairs:
            ids = tokenizer.encode(make_text(q, a))[:MAX_SEQ]
            logits = model(mx.array([ids]))[0]          # (L, V)
            probs = np.array(mx.softmax(logits, axis=-1), copy=False)
            acc += probs[:-1].mean(0)
            n += 1
        return acc / max(n, 1)

    mu_f = mean_dist(forget_pairs)
    mu_r = mean_dist(retain_pairs)
    M = np.stack([mu_f, mu_r])
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    rho = float(S[1] ** 2 / (S[0] ** 2 + S[1] ** 2 + 1e-12))
    return rho


def evaluate(tag, model, tokenizer, ev_forget, ev_retain):
    fa = match_rate(model, tokenizer, ev_forget)
    ra = match_rate(model, tokenizer, ev_retain)
    rho = rii_llm(model, tokenizer, ev_forget, ev_retain)
    mia = mean_nll(model, tokenizer, ev_forget)
    print(f"  [{tag:12s}] forget_acc={fa:5.1f}% retain_acc={ra:5.1f}% RII={rho:.4f} MIA={mia:.3f}")
    return dict(method=tag, forget_acc=fa, retain_acc=ra, rii=rho, mia=mia)


def main():
    t0 = time.time()
    print(f"Device: MPS/MLX | loading TOFU + {MODEL_DIR.split('/')[-1]}")
    ev_forget, ev_retain, tr_retain, tr_full = load_tofu()
    print(f"data: ev_forget={len(ev_forget)} ev_retain={len(ev_retain)} "
          f"tr_retain={len(tr_retain)} tr_full={len(tr_full)}")

    # ---- NoUnlearn: LoRA FT on full (retain + forget) ----
    print("\n[1] NoUnlearn: LoRA FT on full TOFU ...")
    model, tokenizer = load_model()
    add_lora(model)
    finetune_lora(model, tr_full, os.path.join(OUT, "adapter_nounlearn.safetensors"))
    rows = [evaluate("NoUnlearn", model, tokenizer, ev_forget, ev_retain)]

    # ---- FineTune: continue LoRA on retain only ----
    print("\n[2] FineTune: continue LoRA on retain ...")
    finetune_lora(model, tr_retain, os.path.join(OUT, "adapter_finetune.safetensors"))
    rows.append(evaluate("FineTune", model, tokenizer, ev_forget, ev_retain))

    # ---- NegGrad: reload NoUnlearn adapter, gradient ascent on forget ----
    print("\n[3] NegGrad: LoRA gradient ascent on forget ...")
    model, tokenizer = load_model()
    add_lora(model)
    from mlx_lm import load as mlx_load
    import mlx.core as mx
    model = mlx_load(MODEL_DIR, adapter_path=os.path.join(OUT, "adapter_nounlearn.safetensors"))[0]
    neggrad_ascent(model, tokenizer, [make_text(q, a) for q, a in ev_forget],
                   os.path.join(OUT, "adapter_neggrad.safetensors"))
    rows.append(evaluate("NegGrad", model, tokenizer, ev_forget, ev_retain))

    # ---- Retrain oracle: fresh LoRA on retain only from base ----
    print("\n[4] Retrain oracle: LoRA FT on retain only from base ...")
    model, tokenizer = load_model()
    add_lora(model)
    finetune_lora(model, tr_retain, os.path.join(OUT, "adapter_retrain.safetensors"))
    rows.append(evaluate("Retrain", model, tokenizer, ev_forget, ev_retain))

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "forget_acc", "retain_acc", "rii", "mia"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nsaved:", os.path.join(OUT, "results.csv"))
    print("=== summary ===")
    for r in rows:
        print(f"  {r['method']:12s} forget={r['forget_acc']:5.1f}% retain={r['retain_acc']:5.1f}% "
              f"RII={r['rii']:.4f} MIA={r['mia']:.3f}")
    print(f"\nall done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
