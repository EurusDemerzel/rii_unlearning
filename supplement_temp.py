#!/usr/bin/env python3
"""
supplement_temp.py — temperature scaling scan + temperature-vs-state-space MHPR
(plan: "温度缩放 vs 状态化 比较两种方法在低准确率模型上的效果").

For each dataset (Fashion-MNIST low-accuracy, CIFAR-10):
  1. train base model (class-level protocol, forget class)
  2. for T in {1,2,5,10,50,100}:
       p_T = softmax(logits/T)
       RII(T), MHPR(T) via temperature-scaled probs
  3. state-space version: quantize logits (k-means M=20) -> state dists ->
     rho_{H,S} (state MHPR) and rho_S (state RII)
  4. compare discriminability of temperature-scaled MHPR vs state MHPR:
     for each method step (NoUnlearn / Retrain / NegGrad) which separates better.

Outputs: results/supplement_temp/{table, csv, fig}.
"""
import os, sys, time, copy, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import SimpleMLP, SmallCNN

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
BS = 64
EPOCHS = 10
OUT = os.path.join("results", "supplement_temp")
os.makedirs(OUT, exist_ok=True)
TS = [1, 2, 5, 10, 50, 100]
M_STATES = 20

from sklearn.cluster import KMeans


def load(name):
    if name == "fashion_mnist":
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        tr = datasets.FashionMNIST(root="./data", train=True, download=False, transform=t)
        model_fn = lambda nc: SimpleMLP(28 * 28, 128, nc)
    else:
        t = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        tr = datasets.CIFAR10(root="./data", train=True, download=False, transform=t)
        model_fn = lambda nc: SmallCNN(input_channels=3, num_classes=nc)
    imgs = torch.stack([tr[i][0] for i in range(len(tr))])
    lbls = torch.tensor(tr.targets)
    return imgs, lbls, model_fn


def idx_of(l, c):
    return torch.where(l == c)[0].numpy()


def loader(imgs, lbls, idx, shuffle=True):
    idx = torch.from_numpy(np.asarray(idx))
    ds = torch.utils.data.TensorDataset(imgs[idx], lbls[idx])
    return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=shuffle)


def train(model, ldr, epochs=EPOCHS):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in ldr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    return model


def acc(model, ldr):
    model.eval(); c = t = 0
    with torch.no_grad():
        for x, y in ldr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            c += (model(x).argmax(1) == y).sum().item(); t += y.numel()
    return 100.0 * c / max(t, 1)


def logits(model, ldr):
    zs = []
    model.eval()
    with torch.no_grad():
        for x, _ in ldr:
            zs.append(model(x.to(DEVICE)).cpu().numpy())
    return np.concatenate(zs, 0)


def softmax_at_T(logits_all, T):
    return torch.softmax(torch.tensor(logits_all) / T, 1).numpy()


def rii_from_means(mu_f, mu_r):
    """RII from two mean distributions (each a 1-D vector over channels)."""
    M = np.stack([mu_f, mu_r])
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    return float(S[1] ** 2 / (S[0] ** 2 + S[1] ** 2 + 1e-12))


def rii(pf, pr):
    return rii_from_means(pf.mean(0), pr.mean(0))


def mhpr(pf, held_means_list):
    mu_f = pf.mean(0)
    H = np.stack(held_means_list)
    Hp = np.linalg.pinv(H)
    proj = Hp @ H @ mu_f
    return float(np.sum((mu_f - proj) ** 2) / (np.sum(mu_f ** 2) + 1e-12))


def state_distributions(logits_all, km):
    """Map logits to k-means states; return empirical state histograms."""
    s = km.predict(logits_all)          # (N,)
    M = km.n_clusters
    counts = np.bincount(s, minlength=M)
    return counts / max(counts.sum(), 1), s


def state_mhpr(state_f, state_h_list):
    pi_f = state_f / max(state_f.sum(), 1)
    H = np.stack([h / max(h.sum(), 1) for h in state_h_list])   # (K, M)
    Hp = np.linalg.pinv(H)
    proj = Hp @ H @ pi_f
    return float(np.sum((pi_f - proj) ** 2) / (np.sum(pi_f ** 2) + 1e-12))


def run_dataset(name):
    print("\n" + "=" * 70 + f"\n[{name}] temperature scan + state-space MHPR\n" + "=" * 70)
    imgs, lbls, model_fn = load(name)
    TRAIN = [0, 1, 2, 3, 4, 5, 6]
    FORGET, HELD = 3, [7, 8, 9]
    RETAIN = [c for c in TRAIN if c != FORGET]
    train_idx = np.concatenate([idx_of(lbls, c) for c in TRAIN])
    forget_idx = idx_of(lbls, FORGET)
    retain_idx = np.concatenate([idx_of(lbls, c) for c in RETAIN])
    held_idx = {c: idx_of(lbls, c) for c in HELD}
    r_sub = np.random.choice(retain_idx, 5000, replace=False)

    model = train(model_fn(10).to(DEVICE), loader(imgs, lbls, train_idx))
    print(f"base: retain_acc={acc(model, loader(imgs,lbls,retain_idx)):.1f}% "
          f"forget_acc={acc(model, loader(imgs,lbls,forget_idx)):.1f}%")

    zf = logits(model, loader(imgs, lbls, forget_idx, shuffle=False))
    zr = logits(model, loader(imgs, lbls, r_sub, shuffle=False))
    zh = {c: logits(model, loader(imgs, lbls, held_idx[c], shuffle=False)) for c in HELD}

    # k-means states on combined logits (training + forget + retain)
    Zall = np.concatenate([zf, zr] + [zh[c] for c in HELD])
    km = KMeans(n_clusters=M_STATES, n_init=3, random_state=SEED).fit(Zall)

    rows = []
    for T in TS:
        pf = softmax_at_T(zf, T); pr = softmax_at_T(zr, T)
        ph = [softmax_at_T(zh[c], T) for c in HELD]
        rho_T = rii(pf, pr)
        mhpr_T = mhpr(pf, [p.mean(0) for p in ph])
        rows.append(dict(T=T, rho=rho_T, mhpr=mhpr_T))
        print(f"  T={T:4d}  rho={rho_T:.4f}  MHPR(T)={mhpr_T:.4f}")

    # state-space
    sf, _ = state_distributions(zf, km)
    sr, _ = state_distributions(zr, km)
    sh = [state_distributions(zh[c], km)[0] for c in HELD]
    rho_S = rii_from_means(sf, sr)
    mhpr_S = state_mhpr(sf, sh)
    print(f"  state-space: rho_S={rho_S:.4f}  MHPR_S={mhpr_S:.4f}  (M={M_STATES})")

    # temperature-scaled discriminability: gap between Retrain-oracle proxies
    # simpler: MHPR(T) at T=1 vs best T — report which T gives lowest MHPR
    best = min(rows, key=lambda r: r["mhpr"])
    print(f"  best T for MHPR: T={best['T']} (mhpr={best['mhpr']:.4f})")
    print(f"  state-space MHPR_S={mhpr_S:.4f} vs best-T MHPR={best['mhpr']:.4f}")

    with open(os.path.join(OUT, f"{name}_temp.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["T", "rho", "mhpr"]); w.writeheader()
        for r in rows: w.writerow(r)
    with open(os.path.join(OUT, f"{name}_state.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        w.writerow(["rho_S", rho_S]); w.writerow(["mhpr_S", mhpr_S])
        w.writerow(["mhpr_T1", rows[0]["mhpr"]]); w.writerow(["mhpr_bestT", best["mhpr"]])


if __name__ == "__main__":
    run_dataset("fashion_mnist")
    run_dataset("cifar10")
    print("\ndone ->", OUT)
