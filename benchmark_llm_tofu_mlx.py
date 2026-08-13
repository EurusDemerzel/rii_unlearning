#!/usr/bin/env python3
"""
benchmark_llm_tofu_mlx.py — TOFU LLM unlearning on LLaMA-2-7B via MLX + LoRA
(Apple Silicon, MPS).

Model: locuslab/tofu_ft_llama2-7b  (official TOFU fine-tuned LLaMA-2-7B;
       downloaded via `hf download` through the system proxy).
Retrain oracle uses a base LLaMA-2-7B (NousResearch/Llama-2-7b-hf) if present.

Protocol (author-level TOFU, forget set = forget01 authors):
  tofu_ft_llama2-7b (already knows all TOFU authors)
    -> NoUnlearn  : load as-is, nothing forgotten
    -> FineTune   : continue LoRA on retain authors only
    -> NegGrad    : LoRA gradient ASCENT on forget authors
    -> Retrain    : base LLaMA-2-7B + LoRA on retain only (never saw forget)
Metrics: forget/retain answer accuracy (ROUGE-L), RII (2 x V channel matrix on
next-token logits, V = vocab), MIA (mean next-token NLL).
Outputs: results/benchmark_llm_tofu_mlx/results.csv
"""
import os, json, time, csv, random, math
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)

OUT = os.path.join("results", "benchmark_llm_tofu_mlx")
os.makedirs(OUT, exist_ok=True)
DATA_DIR = os.path.join("data", "tofu")
FT_MODEL_DIR = os.path.join("data", "models", "tofu_ft_llama2-7b")
BASE_MODEL_DIR = os.path.join("data", "models", "Llama-2-7b-hf")

MAX_SEQ = 384
LR = 1e-5
FT_STEPS = 60
FT_BATCH = 1
NEG_STEPS = 10
EVAL_FORGET = 20
EVAL_RETAIN = 40
GEN_MAX = 32
ROUGE_THR = 0.25
LORA_LAYERS = 8
LORA_RANK = 8
LORA_SCALE = 20.0


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
    return ev_forget, ev_retain, tr_retain


def load_model(model_dir):
    import mlx_lm
    model, tokenizer = mlx_lm.load(model_dir)
    return model, tokenizer


def add_lora(model):
    """Freeze all, then convert layers to LoRA (conversion unfreezes LoRA only)."""
    model.freeze()
    from mlx_lm.tuner.utils import linear_to_lora_layers
    config = {"rank": LORA_RANK, "scale": LORA_SCALE, "dropout": 0.0}
    linear_to_lora_layers(model, num_layers=LORA_LAYERS, config=config)


def finetune_lora(model, tokenizer, texts, adapter_file, steps=FT_STEPS, lr=LR, batch=FT_BATCH):
    """LoRA fine-tune on `texts`; saves adapter to adapter_file."""
    from mlx_lm.lora import train, TrainingArgs
    from mlx_lm.tuner.datasets import TextDataset, CacheDataset
    from mlx_lm.tuner.trainer import default_loss
    import mlx.optimizers as optim
    opt = optim.AdamW(learning_rate=lr)
    args = TrainingArgs(
        batch_size=batch, iters=steps, max_seq_length=MAX_SEQ,
        steps_per_report=steps, steps_per_eval=steps + 1,
        steps_per_save=steps, adapter_file=adapter_file,
    )
    data = CacheDataset(TextDataset([{"text": t} for t in texts], tokenizer))
    train(model=model, optimizer=opt, train_dataset=data,
          args=args, loss=default_loss)
    return adapter_file


def neggrad_ascent(model, tokenizer, texts, adapter_file, steps=NEG_STEPS, lr=LR * 2):
    """Gradient ascent on LoRA params over forget texts (NegGrad)."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm.tuner.trainer import default_loss
    from mlx.utils import tree_flatten, tree_map

    def loss_fn(m, batch, lengths):
        return default_loss(m, batch, lengths)[0]

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.SGD(learning_rate=lr)
    for it in range(steps):
        text = random.choice(texts)
        ids = tokenizer.encode(text)[:MAX_SEQ]
        batch = mx.array([ids])
        lengths = mx.array([[0, max(len(ids) - 2, 1)]])
        loss, grads = loss_and_grad(model, batch, lengths)
        # ascent: p <- p + lr*grad  ==  SGD descent on negated grads
        opt.update(model, tree_map(lambda g: -g, grads))
        if it % 5 == 0 or it == steps - 1:
            print(f"      neggrad step {it+1}/{steps} loss={float(loss):.3f}")
    mx.save_safetensors(adapter_file, dict(tree_flatten(model.trainable_parameters())))
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
        batch = mx.array([ids])
        lengths = mx.array([[0, max(len(ids) - 2, 1)]])
        losses.append(float(default_loss(model, batch, lengths)[0]))
    return float(np.mean(losses))


def rii_llm(model, tokenizer, forget_pairs, retain_pairs):
    """RII from 2 x V mean next-token output distributions (V = vocab)."""
    import mlx.core as mx
    V = tokenizer.vocab_size

    def mean_dist(pairs):
        acc = np.zeros(V, dtype=np.float64)
        n = 0
        for q, a in pairs:
            ids = tokenizer.encode(make_text(q, a))[:MAX_SEQ]
            logits = model(mx.array([ids]))[0]             # (L, V)
            probs = np.array(mx.softmax(logits, axis=-1).astype(mx.float32),
                             copy=False)
            probs = np.nan_to_num(probs, nan=1.0 / V)      # defensive
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
    print(f"Device: MPS/MLX | LLaMA-2-7B TOFU (ft model: {os.path.basename(FT_MODEL_DIR)})")
    ev_forget, ev_retain, tr_retain = load_tofu()
    print(f"data: ev_forget={len(ev_forget)} ev_retain={len(ev_retain)} tr_retain={len(tr_retain)}")
    assert os.path.isdir(FT_MODEL_DIR), f"missing {FT_MODEL_DIR}"

    # ---- NoUnlearn: load TOFU-finetuned model as-is ----
    print("\n[1] NoUnlearn: tofu_ft_llama2-7b as-is ...")
    model, tokenizer = load_model(FT_MODEL_DIR)
    rows = [evaluate("NoUnlearn", model, tokenizer, ev_forget, ev_retain)]

    # ---- FineTune: continue LoRA on retain only ----
    print("\n[2] FineTune: LoRA continue on retain ...")
    add_lora(model)
    finetune_lora(model, tokenizer,
                  [make_text(q, a) for q, a in tr_retain],
                  os.path.join(OUT, "adapter_finetune.safetensors"))
    rows.append(evaluate("FineTune", model, tokenizer, ev_forget, ev_retain))

    # ---- NegGrad: reload NoUnlearn (fresh zero LoRA), ascent on forget ----
    print("\n[3] NegGrad: LoRA gradient ascent on forget ...")
    model, tokenizer = load_model(FT_MODEL_DIR)
    add_lora(model)                       # zero-initialized LoRA == original model
    neggrad_ascent(model, tokenizer,
                   [make_text(q, a) for q, a in ev_forget],
                   os.path.join(OUT, "adapter_neggrad.safetensors"))
    rows.append(evaluate("NegGrad", model, tokenizer, ev_forget, ev_retain))

    # ---- Retrain oracle (optional): base LLaMA-2-7B + LoRA on retain only ----
    base_ready = (os.path.isdir(BASE_MODEL_DIR)
                  and any(f.endswith(".safetensors")
                          for f in os.listdir(BASE_MODEL_DIR)))
    if base_ready:
        print("\n[4] Retrain oracle: base LLaMA-2-7B + LoRA on retain only ...")
        model, tokenizer = load_model(BASE_MODEL_DIR)
        add_lora(model)
        finetune_lora(model, tokenizer,
                      [make_text(q, a) for q, a in tr_retain],
                      os.path.join(OUT, "adapter_retrain.safetensors"))
        rows.append(evaluate("Retrain", model, tokenizer, ev_forget, ev_retain))
    else:
        print(f"\n[4] Skipping Retrain: base model not found at {BASE_MODEL_DIR}")

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
