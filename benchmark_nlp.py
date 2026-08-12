#!/usr/bin/env python3
"""
benchmark_nlp.py — NLP class-level unlearning demo (AG News + DistilBERT).

Answers "is RII/MHPR only for vision CNNs?" — the same class-level protocol on
text with a Transformer.

Data: data/ag_news/train.parquet (modelscope mirror, 120k, 4 classes).
Model: DistilBERT (downloaded via modelscope AI-ModelScope/distilbert-base-uncased).
Protocol (AG News has 4 classes):
  - Train classes : {0,1,2,3} (World, Sports, Business, SciTech), 4000/class
  - Forget class  : 0 (World)
  - Retain        : {1,2,3}
  - Held-out      : none (only 4 classes) -> RII + MIA only; MHPR needs >K unseen
  - Methods       : NoUnlearn / Retrain / NegGrad / FineTune / KED
  - Metrics       : retain_acc, forget_acc, RII (2x4 channel), MIA-loss AUC

Outputs: results/benchmark_nlp/results.csv + printed table.
"""
import os, sys, time, csv, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

MODEL_DIR = "/Users/peregrine/.cache/modelscope/models/AI-ModelScope--distilbert-base-uncased/snapshots/master"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
OUT = os.path.join("results", "benchmark_nlp")
os.makedirs(OUT, exist_ok=True)
print(f"Device: {DEVICE}")

FORGET = 0
RETAIN = [1, 2, 3]
TRAIN_ALL = [0, 1, 2, 3]
PER_CLASS = 4000          # train subsample per class
EVAL_PER_CLASS = 1500     # eval subsample per class
BS = 32
EPOCHS = 2
LR = 2e-5

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token if tok.eos_token else "[PAD]"

# ---------------------------------------------------------------------------
print("loading AG News ...", flush=True)
df = pd.read_parquet("data/ag_news/train.parquet")
data = {}
for c in TRAIN_ALL:
    sub = df[df["label"] == c].reset_index(drop=True)
    tr = sub.iloc[:PER_CLASS]["text"].tolist()
    ev = sub.iloc[PER_CLASS:PER_CLASS + EVAL_PER_CLASS]["text"].tolist()
    data[c] = dict(train=tr, eval=ev)
print("loaded 4 classes x", PER_CLASS, "train /", EVAL_PER_CLASS, "eval")


def build_loader(texts, labels, shuffle=True):
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    y = torch.tensor(labels)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], y)
    return DataLoader(ds, batch_size=BS, shuffle=shuffle)


train_texts, train_lbls = [], []
for c in TRAIN_ALL:
    train_texts += data[c]["train"]; train_lbls += [c] * PER_CLASS
train_loader = build_loader(train_texts, train_lbls)

forget_texts = data[FORGET]["eval"]
retain_texts = sum([data[c]["eval"] for c in RETAIN], [])
forget_loader = build_loader(forget_texts, [FORGET] * len(forget_texts), shuffle=False)
retain_loader = build_loader(retain_texts, [c for c in RETAIN for _ in range(EVAL_PER_CLASS)], shuffle=False)


def new_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=4).to(DEVICE)


def train(model, loader, epochs=EPOCHS, lr=LR):
    opt = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for ids, mask, y in loader:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(ids, attention_mask=mask, labels=y)
            out.loss.backward(); opt.step()
    return model


def acc(model, loader):
    model.eval(); c = t = 0
    with torch.no_grad():
        for ids, mask, y in loader:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            c += (model(ids, attention_mask=mask).logits.argmax(1) == y).sum().item()
            t += y.numel()
    return 100.0 * c / max(t, 1)


def probs(model, loader):
    ps = []
    model.eval()
    with torch.no_grad():
        for ids, mask, _ in loader:
            ps.append(torch.softmax(model(ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits, 1).cpu().numpy())
    return np.concatenate(ps, 0)


def losses(model, loader):
    out = []
    model.eval()
    with torch.no_grad():
        for ids, mask, y in loader:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            logits = model(ids, attention_mask=mask).logits
            out.append(nn.CrossEntropyLoss(reduction="none")(logits, y).cpu().numpy())
    return np.concatenate(out, 0)


def rii(pf, pr):
    M = np.stack([pf.mean(0), pr.mean(0)])
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    return float(S[1] ** 2 / (S[0] ** 2 + S[1] ** 2 + 1e-12))


def mia_loss_auc(lf, lr):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(lf)), np.zeros(len(lr))])
    s = np.concatenate([-lf, -lr])
    return float(roc_auc_score(y, s))


def grad_ascent(model, steps, lr=1e-5):
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    it = iter(forget_loader)
    for _ in range(steps):
        try:
            ids, mask, y = next(it)
        except StopIteration:
            it = iter(forget_loader); ids, mask, y = next(it)
        ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); (-crit(model(ids, attention_mask=mask).logits, y)).backward(); opt.step()
    return model


def eval_model(model):
    pf = probs(model, forget_loader)
    pr = probs(model, retain_loader)
    lf = losses(model, forget_loader)
    lr = losses(model, retain_loader)
    return dict(retain_acc=acc(model, retain_loader), forget_acc=acc(model, forget_loader),
                rii=rii(pf, pr), mia_loss_auc=mia_loss_auc(lf, lr))


# ---------------------------------------------------------------------------
print("finetuning base DistilBERT ...", flush=True)
t0 = time.time()
base = train(new_model(), train_loader)
print(f"base done {time.time()-t0:.0f}s | retain={acc(base,retain_loader):.1f}% forget={acc(base,forget_loader):.1f}%")

results = {"NoUnlearn": eval_model(base)}

t0 = time.time()
ret = train(new_model(), build_loader(retain_texts, [c for c in RETAIN for _ in range(EVAL_PER_CLASS)]), epochs=1)
results["Retrain"] = eval_model(ret)
print(f"Retrain {time.time()-t0:.0f}s")

def method(name, fn):
    t0 = time.time()
    m = fn()
    results[name] = eval_model(m)
    r = results[name]
    print(f"{name}: {time.time()-t0:.0f}s | retain={r['retain_acc']:.1f}% forget={r['forget_acc']:.1f}% rho={r['rii']:.4f} MIA={r['mia_loss_auc']:.3f}")

def method_ked(model, epochs=1, lr=1e-5):
    m = copy.deepcopy(model)
    opt = optim.Adam(m.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    uniform = torch.full((1, 4), 0.25, device=DEVICE)
    m.train()
    for _ in range(epochs):
        for ids, mask, y in forget_loader:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = m(ids, attention_mask=mask).logits
            kl = nn.functional.kl_div(nn.functional.log_softmax(logits, 1),
                                      uniform.expand(ids.size(0), -1), reduction="batchmean")
            kl.backward(); opt.step()
        for ids, mask, y in retain_loader:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(m(ids, attention_mask=mask).logits, y).backward(); opt.step()
    return m

method("NegGrad", lambda: grad_ascent(copy.deepcopy(base), 60))
method("FineTune", lambda: train(grad_ascent(copy.deepcopy(base), 40), build_loader(retain_texts, [c for c in RETAIN for _ in range(EVAL_PER_CLASS)]), epochs=1))
method("KED", lambda: method_ked(copy.deepcopy(base)))

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print(f"{'method':<11s} {'retain%':>7s} {'forget%':>7s} {'RII':>8s} {'MIA-loss':>9s}")
print("-" * 72)
with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "retain_acc", "forget_acc", "rii", "mia_loss_auc"])
    for name in ["NoUnlearn", "Retrain", "NegGrad", "FineTune", "KED"]:
        r = results[name]
        print(f"{name:<11s} {r['retain_acc']:7.1f} {r['forget_acc']:7.1f} {r['rii']:8.4f} {r['mia_loss_auc']:9.3f}")
        w.writerow([name, f"{r['retain_acc']:.1f}", f"{r['forget_acc']:.1f}", f"{r['rii']:.6f}", f"{r['mia_loss_auc']:.4f}"])
print("saved:", os.path.join(OUT, "results.csv"))
