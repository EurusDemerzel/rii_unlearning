#!/usr/bin/env python3
"""
benchmark_llm_muse.py — MUSE-style long-form book unlearning on LLaMA-2-7B
via MLX + LoRA (Apple Silicon, MPS).

Data (public domain, Project Gutenberg):
  forget = Alice's Adventures in Wonderland  (Lewis Carroll, 1865)
  retain = The Wonderful Wizard of Oz        (L. Frank Baum, 1900)
Built by prepare_muse.py -> data/muse/muse_data.pkl (chunked w/ LLaMA-2 tokenizer,
held-out eval chunks disjoint from training chunks).

Protocol (mirrors MUSE / TOFU):
  teacher   = base LLaMA-2-7B + LoRA on forget (Alice)   [NoUnlearn baseline]
  NoUnlearn = teacher as-is
  FineTune  = teacher LoRA continue-tuned on retain (Oz)
  NegGrad   = teacher LoRA gradient ascent on forget (Alice)
  Retrain   = base LLaMA-2-7B + LoRA on retain only (never saw Alice)

Metrics:
  forget_ppl / retain_ppl  = exp(mean next-token NLL) on held-out chunks
  rii                      = RII from 2 x V mean next-token distributions
  mia                      = mean next-token NLL on forget (= log forget_ppl)
  forget_rouge             = ROUGE-L of 64-token continuation vs. original
Outputs: results/benchmark_llm_muse/results.csv
"""
import os, json, time, csv, random, math
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)

OUT = os.path.join("results", "benchmark_llm_muse")
os.makedirs(OUT, exist_ok=True)
DATA_PKL = os.path.join("data", "muse", "muse_data.pkl")
BASE_MODEL_DIR = os.path.join("data", "models", "Llama-2-7b-hf")

MAX_SEQ = 384
LR = 1e-5
FT_STEPS = 60
FT_BATCH = 1
NEG_STEPS = 15
TCH_STEPS = 200
GEN_MAX = 64
PROMPT_TOK = 40
LORA_LAYERS = 8
LORA_RANK = 8
LORA_SCALE = 20.0
SMOKE = False   # quick sanity run (tiny teacher, tiny eval)


def load_pickle(p):
    import pickle
    with open(p, "rb") as fh:
        return pickle.load(fh)


def load_model(model_dir):
    import mlx_lm
    model, tokenizer = mlx_lm.load(model_dir)
    return model, tokenizer


def add_lora(model):
    model.freeze()
    from mlx_lm.tuner.utils import linear_to_lora_layers
    config = {"rank": LORA_RANK, "scale": LORA_SCALE, "dropout": 0.0}
    linear_to_lora_layers(model, num_layers=LORA_LAYERS, config=config)


def load_teacher_lora(model, adapter_file):
    """Freeze base, build LoRA structure, load teacher LoRA weights from file."""
    import mlx.core as mx
    model.freeze()
    from mlx_lm.tuner.utils import linear_to_lora_layers
    config = {"rank": LORA_RANK, "scale": LORA_SCALE, "dropout": 0.0}
    linear_to_lora_layers(model, num_layers=LORA_LAYERS, config=config)
    w = mx.load(adapter_file)
    model.load_weights(w, strict=False)
    return model


def finetune_lora(model, tokenizer, texts, adapter_file, steps=FT_STEPS, lr=LR, batch=FT_BATCH):
    """LoRA fine-tune via a plain manual loop (mlx_lm.lora.train() is avoided:
    its mx.compile'd step produces NaN losses in this environment)."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm.tuner.trainer import default_loss
    from mlx.utils import tree_flatten

    def loss_fn(m, b, lengths):
        return default_loss(m, b, lengths)[0]

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=lr)
    for it in range(steps):
        text = random.choice(texts)
        ids = tokenizer.encode(text)[:MAX_SEQ]
        b = mx.array([ids])
        lengths = mx.array([[0, max(len(ids) - 2, 1)]])
        loss, grads = loss_and_grad(model, b, lengths)
        opt.update(model, grads)
        mx.eval(model.parameters())
        if it % 10 == 0 or it == steps - 1:
            print(f"    finetune step {it+1}/{steps} loss={float(loss):.3f}")
    mx.save_safetensors(adapter_file, dict(tree_flatten(model.trainable_parameters())))
    return adapter_file


def neggrad_ascent(model, tokenizer, texts, adapter_file, steps=NEG_STEPS, lr=LR * 2):
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


def mean_nll(model, tokenizer, texts):
    import mlx.core as mx
    from mlx_lm.tuner.trainer import default_loss
    losses = []
    for t in texts:
        ids = tokenizer.encode(t)[:MAX_SEQ]
        batch = mx.array([ids])
        lengths = mx.array([[0, max(len(ids) - 2, 1)]])
        losses.append(float(default_loss(model, batch, lengths)[0]))
    return float(np.mean(losses))


def continuation_rouge(model, tokenizer, texts, n=25):
    """ROUGE-L of generated continuation vs. original chunk (MUSE forget quality)."""
    from mlx_lm import generate
    scores = []
    for t in texts[:n]:
        ids = tokenizer.encode(t)
        prompt_ids = ids[:PROMPT_TOK]
        prompt = tokenizer.decode(prompt_ids)
        ref_ids = ids[PROMPT_TOK:PROMPT_TOK + GEN_MAX]
        ref = tokenizer.decode(ref_ids)
        gen = generate(model, tokenizer, prompt=prompt, max_tokens=GEN_MAX)
        scores.append(rouge_l_f1(gen, ref, tokenizer))
    return 100.0 * float(np.mean(scores))


def rii_llm(model, tokenizer, forget_texts, retain_texts):
    import mlx.core as mx
    V = tokenizer.vocab_size

    def mean_dist(texts):
        acc = np.zeros(V, dtype=np.float64)
        n = 0
        for t in texts:
            ids = tokenizer.encode(t)[:MAX_SEQ]
            logits = model(mx.array([ids]))[0]
            probs = np.array(mx.softmax(logits, axis=-1).astype(mx.float32), copy=False)
            probs = np.nan_to_num(probs, nan=1.0 / V)
            acc += probs[:-1].mean(0)
            n += 1
        return acc / max(n, 1)

    mu_f = mean_dist(forget_texts)
    mu_r = mean_dist(retain_texts)
    M = np.stack([mu_f, mu_r])
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    rho = float(S[1] ** 2 / (S[0] ** 2 + S[1] ** 2 + 1e-12))
    return rho


def evaluate(tag, model, tokenizer, ev_forget, ev_retain, do_rouge=True):
    f_nll = mean_nll(model, tokenizer, ev_forget)
    r_nll = mean_nll(model, tokenizer, ev_retain)
    rho = rii_llm(model, tokenizer, ev_forget, ev_retain)
    f_rouge = continuation_rouge(model, tokenizer, ev_forget) if do_rouge else float("nan")
    res = dict(method=tag,
               forget_ppl=math.exp(f_nll), retain_ppl=math.exp(r_nll),
               rii=rho, mia=f_nll, forget_rouge=f_rouge)
    print(f"  [{tag:10s}] forget_ppl={res['forget_ppl']:7.2f} retain_ppl={res['retain_ppl']:7.2f} "
          f"RII={rho:.4f} MIA={f_nll:.3f} forget_rouge={f_rouge:.1f}%")
    return res


def main():
    t0 = time.time()
    print(f"Device: MPS/MLX | LLaMA-2-7B MUSE-style (Alice forget / Oz retain)")
    data = load_pickle(DATA_PKL)
    tr_forget, ev_forget = data["tr_forget"], data["ev_forget"]
    tr_retain, ev_retain = data["tr_retain"], data["ev_retain"]
    if SMOKE:
        tr_forget, ev_forget = tr_forget[:12], ev_forget[:4]
        tr_retain, ev_retain = tr_retain[:16], ev_retain[:4]
    print(f"data: tr_forget={len(tr_forget)} ev_forget={len(ev_forget)} "
          f"tr_retain={len(tr_retain)} ev_retain={len(ev_retain)}")
    assert os.path.isdir(BASE_MODEL_DIR), f"missing {BASE_MODEL_DIR}"

    tch_steps = 12 if SMOKE else TCH_STEPS
    ft_steps = 10 if SMOKE else FT_STEPS
    neg_steps = 5 if SMOKE else NEG_STEPS

    # ---- Teacher: base + LoRA on Alice ----
    print(f"\n[0] Teacher: base + LoRA on forget (Alice), {tch_steps} steps ...")
    model, tokenizer = load_model(BASE_MODEL_DIR)
    add_lora(model)
    teacher_adapter = os.path.join(OUT, "adapter_teacher.safetensors")
    finetune_lora(model, tokenizer, tr_forget, teacher_adapter, steps=tch_steps)
    rows = [evaluate("NoUnlearn", model, tokenizer, ev_forget, ev_retain)]

    # ---- FineTune: continue teacher LoRA on retain ----
    print(f"\n[1] FineTune: teacher LoRA continue on retain (Oz), {ft_steps} steps ...")
    finetune_lora(model, tokenizer, tr_retain,
                  os.path.join(OUT, "adapter_finetune.safetensors"), steps=ft_steps)
    rows.append(evaluate("FineTune", model, tokenizer, ev_forget, ev_retain))

    # ---- NegGrad: reload teacher + ascent on forget ----
    print(f"\n[2] NegGrad: teacher LoRA gradient ascent on forget, {neg_steps} steps ...")
    model, tokenizer = load_model(BASE_MODEL_DIR)
    load_teacher_lora(model, teacher_adapter)
    neggrad_ascent(model, tokenizer, tr_forget,
                   os.path.join(OUT, "adapter_neggrad.safetensors"), steps=neg_steps)
    rows.append(evaluate("NegGrad", model, tokenizer, ev_forget, ev_retain))

    # ---- Retrain oracle: base + LoRA on retain only ----
    print(f"\n[3] Retrain oracle: base + LoRA on retain only, {ft_steps} steps ...")
    model, tokenizer = load_model(BASE_MODEL_DIR)
    add_lora(model)
    finetune_lora(model, tokenizer, tr_retain,
                  os.path.join(OUT, "adapter_retrain.safetensors"), steps=ft_steps)
    rows.append(evaluate("Retrain", model, tokenizer, ev_forget, ev_retain))

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "forget_ppl", "retain_ppl", "rii", "mia", "forget_rouge"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nsaved:", os.path.join(OUT, "results.csv"))
    print("=== summary ===")
    for r in rows:
        print(f"  {r['method']:10s} forget_ppl={r['forget_ppl']:7.2f} retain_ppl={r['retain_ppl']:7.2f} "
              f"RII={r['rii']:.4f} MIA={r['mia']:.3f} rouge={r['forget_rouge']:.1f}%")
    print(f"\nall done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
