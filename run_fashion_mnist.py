#!/usr/bin/env python3
"""
Fashion-MNIST experiment suite: random forgetting, class-level forgetting,
multi-class projection residual, gradient sweep.

Fashion-MNIST: 60K train / 10K test, 28×28 grayscale, 10 classes.
Same format as MNIST but with clothing items (harder task: ~92% vs ~99%).
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model
from metrics import compute_rii_from_probs, compute_channel_rank

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")
OUT_DIR = "results/fashion_mnist"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 128
LR = 0.001
EPOCHS = 20
FORGET_RATIO = 0.05
GRADIENT_EPOCHS = [0, 1, 3, 10, 30, 100]
GRADIENT_LR = 1e-5

# ─── Load Fashion-MNIST ───
print("Loading Fashion-MNIST...")
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=tfm)
test_set  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=tfm)

all_imgs = torch.stack([train_set[i][0] for i in range(len(train_set))]).squeeze(1)  # 60K×28×28
all_lbls = torch.tensor([train_set[i][1] for i in range(len(train_set))])
test_imgs = torch.stack([test_set[i][0] for i in range(len(test_set))]).squeeze(1).to(DEVICE)
test_lbls = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
C = 10
print(f"  Train: {len(all_imgs)}  Test: {len(test_imgs)}  Classes: {C}")
train_loader = DataLoader(TensorDataset(all_imgs, all_lbls), batch_size=BATCH_SIZE, shuffle=True)

# ─── Train SimpleMLP ───
print(f"\nTraining SimpleMLP {EPOCHS} epochs...")
m = get_model('mnist', DEVICE, model_name='mlp')
opt = optim.Adam(m.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    m.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = (m(test_imgs).argmax(1) == test_lbls).float().mean().item()
    print(f"  Ep {ep:2d}/{EPOCHS} | acc={acc*100:.2f}% | {time.time()-t0:.0f}s")
total_time = time.time() - t0
final_acc = acc
print(f"Training done in {total_time:.0f}s, final acc={final_acc*100:.2f}%")
torch.save(m.state_dict(), os.path.join(OUT_DIR, 'checkpoint_full.pt'))

# ═══════════════════════════════════════════════════
#  Experiment 1: Random Forgetting (5%)
# ═══════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Experiment 1: Random Forgetting (5%)")
print(f"{'='*60}")
N = len(all_imgs)
N_f = int(N * FORGET_RATIO)
perm = torch.randperm(N)
forget_idx = perm[:N_f].tolist()
retain_idx = perm[N_f:].tolist()

with torch.no_grad():
    pf = torch.softmax(m(all_imgs[forget_idx].to(DEVICE)), 1).cpu().numpy()
    pr = torch.softmax(m(all_imgs[retain_idx].to(DEVICE)), 1).cpu().numpy()
rho_random, mi_ub = compute_rii_from_probs(pf, pr)
sr, _ = compute_channel_rank(pf, pr)
print(f"  Random forget (N_f={N_f}): ρ={rho_random:.6f} | σ₂/σ₁={sr:.6f}")

# ═══════════════════════════════════════════════════
#  Experiment 2: Class-Level Forgetting
# ═══════════════════════════════════════════════════
FORGET_CLASS = 5  # "Sandal" — one of the 10 classes
print(f"\n{'='*60}")
print(f"Experiment 2: Class-Level Forgetting (class {FORGET_CLASS})")
print(f"{'='*60}")

class_idx = (all_lbls == FORGET_CLASS).nonzero().squeeze().tolist()
if isinstance(class_idx, int): class_idx = [class_idx]
retain_idx = [i for i in range(N) if i not in class_idx]
print(f"  Forget class {FORGET_CLASS}: {len(class_idx)} samples")

with torch.no_grad():
    pf = torch.softmax(m(all_imgs[class_idx].to(DEVICE)), 1).cpu().numpy()
    pr = torch.softmax(m(all_imgs[retain_idx].to(DEVICE)), 1).cpu().numpy()
rho_class, _ = compute_rii_from_probs(pf, pr)
sr_class, _ = compute_channel_rank(pf, pr)
print(f"  Class forget: ρ={rho_class:.6f} | σ₂/σ₁={sr_class:.6f}")
print(f"  Ratio class/random: {rho_class/rho_random:.1f}×")

# ═══════════════════════════════════════════════════
#  Experiment 3: MHPR (Multi-Class Projection)
# ═══════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Experiment 3: Multi-Held-Out Projection Residual")
print(f"{'='*60}")

heldout_candidates = [c for c in range(C) if c != FORGET_CLASS]
def mean_softmax(indices):
    with torch.no_grad():
        x = all_imgs[indices].to(DEVICE)
        return torch.softmax(m(x), 1).mean(0).cpu().numpy()

mu_f = mean_softmax(class_idx)
proj_results = {}
for K in [1, 2, 3, 5, 9]:
    H_rows = []
    for c in heldout_candidates[:K]:
        c_idx = (all_lbls == c).nonzero().squeeze().tolist()
        if isinstance(c_idx, int): c_idx = [c_idx]
        H_rows.append(mean_softmax(c_idx))
    H = np.stack(H_rows)
    H_plus = np.linalg.pinv(H)
    residual = mu_f - mu_f @ H_plus @ H
    rho_H = float(np.sum(residual**2) / (np.sum(mu_f**2) + 1e-12))
    proj_results[K] = rho_H
    print(f"  K={K} | ρ_H={rho_H:.6f}")

# ═══════════════════════════════════════════════════
#  Experiment 4: Gradient Sweep (Class-Level)
# ═══════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Experiment 4: Gradient Sweep (Class-Level)")
print(f"{'='*60}")

f_imgs, f_lbls = all_imgs[class_idx].to(DEVICE), all_lbls[class_idx].to(DEVICE)
r_imgs, r_lbls = all_imgs[retain_idx].to(DEVICE), all_lbls[retain_idx].to(DEVICE)
n_f = len(class_idx)

sweep_results = []
state_dict = torch.load(os.path.join(OUT_DIR, 'checkpoint_full.pt'), map_location=DEVICE)

for target in GRADIENT_EPOCHS:
    m2 = get_model('mnist', DEVICE, model_name='mlp')
    m2.load_state_dict(state_dict)
    if target > 0:
        opt2 = optim.Adam(m2.parameters(), lr=GRADIENT_LR)
        m2.train()
        for _ in range(target):
            perm = torch.randperm(n_f)
            for i in range(0, n_f, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt2.zero_grad()
                loss = -crit(m2(f_imgs[idx]), f_lbls[idx])
                loss.backward()
                opt2.step()
    m2.eval()
    with torch.no_grad():
        pf = torch.softmax(m2(f_imgs), 1).cpu().numpy()
        pr = torch.softmax(m2(r_imgs), 1).cpu().numpy()
        acc = (m2(test_imgs).argmax(1) == test_lbls).float().mean().item()
    rho, _ = compute_rii_from_probs(pf, pr)
    sr, _ = compute_channel_rank(pf, pr)
    sweep_results.append(dict(epochs=target, rho=float(rho), sigma_ratio=float(sr), test_acc=float(acc)))
    print(f"  epochs={target:3d} | ρ={rho:.6f} | σ₂/σ₁={sr:.6f} | acc={acc*100:.2f}%")

# ═══════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════
res = dict(
    dataset="Fashion-MNIST",
    model="SimpleMLP",
    num_classes=C,
    epochs=EPOCHS,
    final_acc=float(final_acc),
    random_forgetting=dict(rho=float(rho_random), sigma_ratio=float(sr), forget_ratio=FORGET_RATIO),
    class_forgetting=dict(forget_class=FORGET_CLASS, rho=float(rho_class), sigma_ratio=float(sr_class)),
    mhpr=proj_results,
    gradient_sweep=sweep_results,
    time_sec=total_time,
)
path = os.path.join(OUT_DIR, 'result.json')
with open(path, 'w') as f:
    json.dump(res, f, indent=2)
print(f"\nSaved to {path}")
print("Done!")
