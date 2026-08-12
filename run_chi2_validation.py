#!/usr/bin/env python3
"""Validate Theorem B: MHPR ≤ 2χ²/||π_f||², M=4C states, K=3 heldout."""
import os, sys, json
import numpy as np
import torch, torch.nn as nn
from torchvision import datasets, transforms
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─── Load MNIST ───
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST(root='./data', train=True, download=False, transform=tfm)
all_imgs = torch.stack([train_set[i][0] for i in range(len(train_set))]).squeeze(1)
all_lbls = torch.tensor([train_set[i][1] for i in range(len(train_set))])
C, M = 10, 20  # 2×C states

# ─── Train SimpleMLP ───
m = get_model('mnist', DEVICE, model_name='mlp')
opt = torch.optim.Adam(m.parameters(), lr=0.001)
crit = nn.CrossEntropyLoss()
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(all_imgs, all_lbls), batch_size=128, shuffle=True)
for ep in range(5):
    m.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
m.eval()
print("Model trained")

# ─── Extract all logits and quantize (M=40) ───
all_logits = []
with torch.no_grad():
    for i in range(0, len(all_imgs), 256):
        batch = all_imgs[i:i+256].to(DEVICE)
        all_logits.append(m(batch).cpu().numpy())
all_logits = np.concatenate(all_logits)

kmeans = MiniBatchKMeans(n_clusters=M, random_state=42, batch_size=1024)
states = kmeans.fit_predict(all_logits)
print(f"Quantization done, M={M}")

def state_dist(labels_subset):
    s = states[labels_subset]
    hist = np.bincount(s, minlength=M).astype(float)
    return hist / hist.sum()

def project_mhpr(pi_f, H):
    """ρ_H = ||residual||² / ||π_f||²."""
    K = H.shape[0]
    H_Ht = H @ H.T
    H_Ht_inv = np.linalg.inv(H_Ht + 1e-12 * np.eye(K))
    alpha = H_Ht_inv @ (H @ pi_f)
    mu_hat = H.T @ alpha
    residual = pi_f - mu_hat
    return float(np.sum(residual**2) / (np.sum(pi_f**2) + 1e-12))

def chi_sq(pi_f, pi_h):
    eps = 1e-12
    return float(np.sum((pi_f - pi_h)**2 / (pi_h + eps)))

# ─── Gradient-ascent sweep ───
FORGET_CLASS = 5
heldout = [c for c in range(C) if c != FORGET_CLASS][:3]  # K=3
idx_f = (all_lbls == FORGET_CLASS).numpy().nonzero()[0]

results = []
for steps in [0, 1, 3, 5, 10, 20, 50, 100]:
    m2 = get_model('mnist', DEVICE, model_name='mlp')
    m2.load_state_dict(m.state_dict())
    if steps > 0:
        opt2 = torch.optim.Adam(m2.parameters(), lr=1e-5)
        f_imgs = all_imgs[idx_f].to(DEVICE)
        f_lbls = all_lbls[idx_f].to(DEVICE)
        m2.train()
        for _ in range(steps):
            opt2.zero_grad()
            loss = -crit(m2(f_imgs), f_lbls)
            loss.backward(); opt2.step()
    m2.eval()

    pl = []
    with torch.no_grad():
        for i in range(0, len(all_imgs), 256):
            batch = all_imgs[i:i+256].to(DEVICE)
            pl.append(m2(batch).cpu().numpy())
    pl = np.concatenate(pl)
    ps = kmeans.predict(pl)

    pi_f_pert = state_dist(ps[idx_f])
    pi_h_pert = np.stack([state_dist(ps[(all_lbls == c).numpy().nonzero()[0]]) for c in heldout])

    rho_H = project_mhpr(pi_f_pert, pi_h_pert)
    a = pi_f_pert
    b = pi_h_pert.mean(axis=0)
    chi2 = chi_sq(a, b)
    norm_f_sq = np.sum(pi_f_pert**2)
    bound = 2 * chi2 / max(norm_f_sq, 1e-12)
    ok = rho_H <= bound + 1e-10
    results.append(dict(steps=steps, rho_H=rho_H, chi2=chi2, norm_f_sq=norm_f_sq, bound=bound))
    print(f"steps={steps:3d} | ρ_H={rho_H:.6f} | χ²={chi2:.6f} | 2χ²/||π||²={bound:.6f} | {'✓' if ok else '✗'}")

path = "results/mhpr_review/chi2_validation.json"
os.makedirs("results/mhpr_review", exist_ok=True)
with open(path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {path}")
