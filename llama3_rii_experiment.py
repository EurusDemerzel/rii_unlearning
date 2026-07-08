#!/usr/bin/env python3
"""RII Validation on LLaMA 3.2 1B via MPS. Vocab ~128K. IMDB pos vs neg."""

import numpy as np
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── 第1部分: 加载模型 ───────────────────────────────────────────
print("=" * 60)
print("Part 1: Loading Qwen2.5 0.5B on MPS")
print("=" * 60)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "mps" if torch.backends.mps.is_available() else "cpu"
from modelscope import snapshot_download
import os

model_id = "qwen/Qwen2.5-0.5B"
cache_dir = os.path.expanduser("~/.cache/modelscope")

print(f"Model: {model_id}")
print(f"Device: {device}")
print(f"Downloading from modelscope...")
model_path = snapshot_download(model_id, cache_dir=cache_dir)
print(f"Model at: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading model (fp16, ~2GB)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map=device,
    trust_remote_code=True,
)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s")
print(f"Vocab size: {model.config.vocab_size:,}")

# ─── 第2部分: 加载数据 ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Part 2: Loading IMDB dataset")
print("=" * 60)

from datasets import load_dataset
dataset = load_dataset("stanfordnlp/imdb", split="train")
print(f"Total: {len(dataset)}")

N_PER_GROUP = 2000

pos_texts = [s["text"] for s in dataset if s["label"] == 1 and len(s["text"]) > 100]
neg_texts = [s["text"] for s in dataset if s["label"] == 0 and len(s["text"]) > 100]

import random
random.seed(42)
pos_texts = random.sample(pos_texts, N_PER_GROUP)
neg_texts = random.sample(neg_texts, N_PER_GROUP)

def make_prompt(text, max_chars=60):
    return text[:max_chars].rsplit(" ", 1)[0]

group_pos = [make_prompt(t) for t in pos_texts]
group_neg = [make_prompt(t) for t in neg_texts]

all_texts = group_pos + group_neg
random.shuffle(all_texts)
group_mixed_1 = all_texts[:N_PER_GROUP]
group_mixed_2 = all_texts[N_PER_GROUP:2*N_PER_GROUP]

print(f"Pos={len(group_pos)}, Neg={len(group_neg)}, "
      f"Mixed1={len(group_mixed_1)}, Mixed2={len(group_mixed_2)}")

# ─── 第3部分: 批量提取概率 ───────────────────────────────────────
print("\n" + "=" * 60)
print("Part 3: Next-token probability extraction")
print("=" * 60)

@torch.no_grad()
def get_next_token_probs(prompts, model, tokenizer, batch_size=8, max_len=64):
    """批量提取下一个token的概率分布"""
    all_probs = []
    n = (len(prompts) + batch_size - 1) // batch_size

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        logits = model(**inputs).logits          # (B, L, V)
        last_logits = logits[:, -1, :]            # (B, V)
        probs = torch.softmax(last_logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())

        if (i//batch_size) % 25 == 0:
            print(f"  {i//batch_size+1}/{n} ({len(all_probs)*batch_size})")

    return np.concatenate(all_probs, axis=0)

# ─── 第4部分: 计算RII ────────────────────────────────────────────
def compute_rii(probs1, probs2):
    mu1 = probs1.mean(axis=0)
    mu2 = probs2.mean(axis=0)
    M = np.stack([mu1, mu2])
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    rho = S[1]**2 / (S[0]**2 + S[1]**2)
    return rho, S[1]/S[0]

# ─── 第5部分: 主实验 ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Part 5: Experiment")
print("=" * 60)

BATCH_SIZE = 4
USE_SUBSET = True   # True=快速(200样本,~3min), False=完整(2000样本,~30min)

if USE_SUBSET:
    N = 200
    group_pos = group_pos[:N]
    group_neg = group_neg[:N]
    group_mixed_1 = group_mixed_1[:N]
    group_mixed_2 = group_mixed_2[:N]
    print(f"SUBSET MODE: {N}/group")

print("[1/3] Positive...")
t0 = time.time()
p_pos = get_next_token_probs(group_pos, model, tokenizer, BATCH_SIZE)
print(f"  {time.time()-t0:.0f}s  shape={p_pos.shape}")

print("[2/3] Negative...")
t0 = time.time()
p_neg = get_next_token_probs(group_neg, model, tokenizer, BATCH_SIZE)
print(f"  {time.time()-t0:.0f}s  shape={p_neg.shape}")

print("[3/3] Mixed...")
t0 = time.time()
p_m1 = get_next_token_probs(group_mixed_1, model, tokenizer, BATCH_SIZE)
p_m2 = get_next_token_probs(group_mixed_2, model, tokenizer, BATCH_SIZE)
print(f"  {time.time()-t0:.0f}s")

# ─── 结果 ────────────────────────────────────────────────────────
rho_pn, sr_pn = compute_rii(p_pos, p_neg)
h = len(p_pos)//2
rho_pp, sr_pp = compute_rii(p_pos[:h], p_pos[h:])
rho_mm, sr_mm = compute_rii(p_m1, p_m2)

V = p_pos.shape[1]
N = len(group_pos)
print(f"\n{'='*60}")
print(f"RESULTS  (C={V:,}, N={N}/group)")
print(f"{'='*60}")
print(f"  rho(pos vs neg):    {rho_pn:.6e}  sigma2/s1={sr_pn:.4f}")
print(f"  rho(pos self-cons): {rho_pp:.6e}  sigma2/s1={sr_pp:.4f}")
print(f"  rho(mixed 1 vs 2):  {rho_mm:.6e}  sigma2/s1={sr_mm:.4f}")
print(f"  Signal/Noise:       {rho_pn/max(rho_pp,rho_mm,1e-15):.1f}x")

if rho_pn > max(rho_pp, rho_mm) * 5:
    print(f"  >> PASS: RII distinguishes semantic groups!")
else:
    print(f"  >> WEAK: low discrimination")

import csv
with open("/Users/peregrine/one_rank/llama3_rii_results.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["Experiment","Group1","Group2","N","C","rho","sigma2/s1"])
    w.writerow(["Semantic diff","Pos","Neg",N,V,f"{rho_pn:.6e}",f"{sr_pn:.6e}"])
    w.writerow(["Self-consistency","Pos-h1","Pos-h2",h,V,f"{rho_pp:.6e}",f"{sr_pp:.6e}"])
    w.writerow(["Null distribution","Mixed1","Mixed2",N,V,f"{rho_mm:.6e}",f"{sr_mm:.6e}"])
print("\nSaved.")
print("DONE!")
