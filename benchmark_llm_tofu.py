#!/usr/bin/env python3
"""
benchmark_llm_tofu.py — real-LLM class-level unlearning on TOFU (Qwen2.5-0.5B).

Validates the RII (Residual Irreversibility Index) protocol on a genuine
large language model, using the TOFU benchmark (locuslab/TOFU, mirrored on
modelscope as popatry/TOFU).

Protocol:
  1. Load Qwen2.5-0.5B (modelscope mirror) + tokenizer.
  2. Fine-tune the base model on a subset of TOFU retain90 authors
     -> the model now "knows" the synthetic authors.
  3. Unlearn methods applied on the forget author set (forget01):
       - NoUnlearn (the fine-tuned model, nothing forgotten)
       - NegGrad   (gradient ascent on forget QA pairs)
       - FineTune  (continued fine-tuning on retain QA pairs)
  4. Metrics per method:
       - retain_acc / forget_acc : token-level answer-match rate on generated answers
       - RII  : 2 x V channel matrix built from mean next-token output distributions
                (V = vocab size) over forget vs retain questions
       - MIA  : average next-token NLL on forget questions
Outputs: results/benchmark_llm_tofu/results.csv

Hardware: Apple M5 Pro 24GB (MPS). Seed 42.
"""
import os, sys, json, time, csv, random
import numpy as np
import torch
import torch.nn as nn
import requests

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

OUT = os.path.join("results", "benchmark_llm_tofu")
os.makedirs(OUT, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
TOFU_BASE = "https://modelscope.cn/datasets/popatry/TOFU/resolve/master"
DATA_DIR = os.path.join("data", "tofu")
os.makedirs(DATA_DIR, exist_ok=True)

MAX_SEQ = 160          # truncation
GEN_MAX = 32           # generated answer tokens
FT_STEPS = 40          # fine-tune steps (limited for wall-clock)
NEG_STEPS = 8          # gradient-ascent steps
EVAL_FORGET = 20       # forget questions used for evaluation
EVAL_RETAIN = 40       # retain questions used for evaluation
LR = 5e-5


def read_jsonl(path):
    """TOFU files on modelscope are JSONL: one {question, answer} object per line."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def download_tofu():
    """Download the small TOFU json files from modelscope if not present."""
    needed = ["forget01.json", "retain90.json"]
    for f in needed:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            continue
        url = f"{TOFU_BASE}/{f}"
        print(f"downloading {f} ...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        open(p, "wb").write(r.content)
    forget = read_jsonl(os.path.join(DATA_DIR, "forget01.json"))
    retain = read_jsonl(os.path.join(DATA_DIR, "retain90.json"))
    print(f"TOFU: forget01={len(forget)} QA  retain90={len(retain)} QA")
    return forget, retain


def make_prompt(q):
    return f"Question: {q}\n\nAnswer:"


def load_model():
    from modelscope import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_dir = snapshot_download(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float32)
    model.to(DEVICE)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def collate(pairs, tok, shuffle=True):
    """Tokenize (q, a) pairs as '<q> <a>' causal-LM batches."""
    texts = [make_prompt(q) + " " + a for q, a in pairs]
    enc = tok(texts, return_tensors="pt", padding="longest",
              truncation=True, max_length=MAX_SEQ)
    return enc


def ft_step(model, tok, pairs, opt, steps, ascent=False, batch_size=4):
    """Mini-batched causal-LM fine-tune / gradient-ascent on (q,a) pairs."""
    model.train()
    total, n = 0.0, 0
    for it in range(steps):
        batch = random.sample(pairs, min(batch_size, len(pairs)))
        enc = collate(batch, tok)
        input_ids = enc["input_ids"].to(DEVICE)
        attn = enc["attention_mask"].to(DEVICE)
        # causal LM labels: predict next token everywhere except padding
        labels = input_ids.clone()
        labels[attn == 0] = -100
        opt.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        if ascent:
            (-loss).backward()          # gradient ascent
        else:
            loss.backward()
        opt.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def generate_answer(model, tok, q, max_new=GEN_MAX):
    prompt = make_prompt(q)
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(**inp, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
    gen = out_ids[0][inp["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def rouge_l_f1(gen, ref, tok):
    """Token-level ROUGE-L F1 between generated and reference answer."""
    g = tok.tokenize(gen.lower())
    r = tok.tokenize(ref.lower())
    m, n = len(g), len(r)
    if m == 0 or n == 0:
        return 0.0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if g[i - 1] == r[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[n]
    prec, rec = lcs / m, lcs / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def match_rate(model, tok, pairs, rouge_thr=0.25):
    """Fraction of questions answered correctly (ROUGE-L F1 >= threshold).

    A stricter match than single-token overlap: many TOFU answers are verbatim
    recoverable from the question (e.g. names), so single-token overlap
    inflates 'accuracy' even for models that never saw the author.
    """
    hits = 0
    with torch.no_grad():
        for q, a in pairs:
            g = generate_answer(model, tok, q)
            if rouge_l_f1(g, a, tok) >= rouge_thr:
                hits += 1
    return 100.0 * hits / max(len(pairs), 1)


def mean_nll(model, tok, pairs, batch_size=4):
    """Average next-token NLL over (q,a) pairs, evaluated in mini-batches."""
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            enc = collate(batch, tok)
            input_ids = enc["input_ids"].to(DEVICE)
            attn = enc["attention_mask"].to(DEVICE)
            labels = input_ids.clone()
            labels[attn == 0] = -100
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            tot += float(out.loss.item()) * len(batch)
            n += len(batch)
    return tot / max(n, 1)


def rii_llm(model, tok, forget_pairs, retain_pairs):
    """RII from a 2 x V channel matrix of mean next-token output distributions.

    For each question we average the softmax next-token distribution over the
    answer span; μ_f and μ_r are the means over forget/retain questions.
    """
    model.eval()
    V = model.config.vocab_size

    def mean_dist(pairs):
        acc = np.zeros(V, dtype=np.float64)
        n = 0
        with torch.no_grad():
            for q, a in pairs:
                text = make_prompt(q) + " " + a
                enc = tok(text, return_tensors="pt").to(DEVICE)
                ids = enc["input_ids"][0]
                if ids.numel() > MAX_SEQ:
                    ids = ids[:MAX_SEQ]
                out = model(input_ids=ids.unsqueeze(0))
                logits = out.logits[0]                      # (L, V)
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                # average over all next-token positions (causal distribution)
                acc += probs[:-1].mean(0)
                n += 1
        return acc / max(n, 1)

    mu_f = mean_dist(forget_pairs)
    mu_r = mean_dist(retain_pairs)
    M = np.stack([mu_f, mu_r])                              # (2, V)
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    rho = float(S[1] ** 2 / (S[0] ** 2 + S[1] ** 2 + 1e-12))
    return rho


def main():
    print(f"Device: {DEVICE} | loading TOFU + Qwen2.5-0.5B ...")
    forget_all, retain_all = download_tofu()

    # --- data split -------------------------------------------------------
    random.shuffle(forget_all); random.shuffle(retain_all)
    forget_pairs = [(d["question"], d["answer"]) for d in forget_all]
    retain_pairs = [(d["question"], d["answer"]) for d in retain_all]
    # disjoint eval / train split on retain (avoid leakage inflating retain_acc)
    ev_retain = retain_pairs[:EVAL_RETAIN]
    tr_retain = retain_pairs[EVAL_RETAIN:EVAL_RETAIN + 240]
    ev_forget = forget_pairs[:EVAL_FORGET]

    model, tok = load_model()
    # snapshot of the untouched base weights (for the Retrain oracle)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # --- 1) fine-tune base on TOFU retain subset -> 'full' model ----------
    print("\n[1] fine-tuning base on TOFU retain subset ...")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    avg = ft_step(model, tok, tr_retain, opt, FT_STEPS, ascent=False)
    print(f"    ft loss={avg:.4f}")

    def evaluate(tag, m):
        fa = match_rate(m, tok, ev_forget)
        ra = match_rate(m, tok, ev_retain)
        rho = rii_llm(m, tok, ev_forget, ev_retain)
        mia = mean_nll(m, tok, ev_forget)
        print(f"  [{tag:12s}] forget_acc={fa:5.1f}% retain_acc={ra:5.1f}% "
              f"RII={rho:.4f} MIA={mia:.4f}")
        return dict(method=tag, forget_acc=fa, retain_acc=ra, rii=rho, mia=mia)

    rows = [evaluate("NoUnlearn", model)]
    # snapshot of the fine-tuned (full-knowledge) model
    ft_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # --- 2) FineTune (continue on retain, starting from NoUnlearn) --------
    print("\n[2] FineTune on retain set ...")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    avg = ft_step(model, tok, tr_retain, opt, FT_STEPS, ascent=False)
    print(f"    ft loss={avg:.4f}")
    rows.append(evaluate("FineTune", model))

    # --- 3) NegGrad (gradient ascent on forget, from NoUnlearn) -----------
    print("\n[3] NegGrad on forget set ...")
    model.load_state_dict({k: v.to(DEVICE) for k, v in ft_state.items()})
    opt = torch.optim.AdamW(model.parameters(), lr=LR * 2)
    avg = ft_step(model, tok, ev_forget, opt, NEG_STEPS, ascent=True)
    print(f"    ascent loss={avg:.4f}")
    rows.append(evaluate("NegGrad", model))

    # --- 4) Retrain oracle (fresh base, retain-only: never saw forget) ----
    print("\n[4] Retrain oracle (base -> retain only, never saw forget) ...")
    model.load_state_dict({k: v.to(DEVICE) for k, v in base_state.items()})
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    avg = ft_step(model, tok, tr_retain, opt, FT_STEPS, ascent=False)
    print(f"    ft loss={avg:.4f}")
    rows.append(evaluate("Retrain", model))

    # --- write results -----------------------------------------------------
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "forget_acc", "retain_acc", "rii", "mia"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nsaved:", os.path.join(OUT, "results.csv"))
    print("\n=== summary ===")
    for r in rows:
        print(f"  {r['method']:12s} forget={r['forget_acc']:5.1f}% retain={r['retain_acc']:5.1f}% "
              f"RII={r['rii']:.4f} MIA={r['mia']:.4f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nall done in {time.time()-t0:.0f}s")
