#!/usr/bin/env python3
"""
CIFAR-100 long training (200 epochs) + gradient sweep + projection residual.
Uses MPS with periodic cache clearing to avoid OOM.

Data is loaded via torchvision.datasets.CIFAR100.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model
from metrics import compute_rii_from_probs, compute_channel_rank

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

OUT_DIR = "results/cifar100_long"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 128
LR = 0.001
EPOCHS = 200
SAVE_EVERY = 50
GRADIENT_LR = 1e-5
GRADIENT_EPOCHS = [0, 1, 3, 10, 30, 100]

# ─── Load CIFAR-100 via torchvision ───
print("Loading CIFAR-100 via torchvision...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_set  = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

all_imgs = torch.stack([train_set[i][0] for i in range(len(train_set))])
all_lbls = torch.tensor([train_set[i][1] for i in range(len(train_set))])
test_imgs = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(DEVICE)
test_lbls = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(DEVICE)
print(f"  Train: {len(all_imgs)}  Test: {len(test_imgs)}  Classes: 100")

# ─── Full training ───
print(f"\nTraining SmallCNN on all 50K samples, {EPOCHS} epochs...")
model = get_model('cifar100', DEVICE, model_name='cnn')
opt = optim.Adam(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()
train_loader = DataLoader(TensorDataset(all_imgs, all_lbls), batch_size=BATCH_SIZE, shuffle=True)

t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    
    # Clear MPS cache every 10 epochs
    if ep % 10 == 0 and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(test_imgs).argmax(1)
        acc = (preds == test_lbls).float().mean().item()
    print(f"  Ep {ep:4d}/{EPOCHS} | acc={acc*100:.2f}% | {time.time()-t0:.0f}s")

    # Save checkpoint
    if ep % SAVE_EVERY == 0 or ep == EPOCHS:
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f'checkpoint_{ep}ep.pt'))

print(f"\nTraining done in {time.time()-t0:.0f}s")

# ─── Gradient sweep on one class ───
FORGET_CLASS = 0
forget_idx = (all_lbls == FORGET_CLASS).nonzero().squeeze().tolist()
if isinstance(forget_idx, int): forget_idx = [forget_idx]
retain_idx = [i for i in range(len(all_imgs)) if i not in forget_idx]

f_imgs, f_lbls = all_imgs[forget_idx].to(DEVICE), all_lbls[forget_idx].to(DEVICE)
r_imgs, r_lbls = all_imgs[retain_idx].to(DEVICE), all_lbls[retain_idx].to(DEVICE)
n_f = len(forget_idx)

print(f"\nGradient sweep (forget class {FORGET_CLASS})...")
m = get_model('cifar100', DEVICE, model_name='cnn')
m.load_state_dict(torch.load(os.path.join(OUT_DIR, 'checkpoint_200ep.pt')))
sweep_opt = optim.Adam(m.parameters(), lr=GRADIENT_LR)
sweep_results = []

for target in GRADIENT_EPOCHS:
    m.train()
    for _ in range(target):
        perm = torch.randperm(n_f)
        for i in range(0, n_f, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            sweep_opt.zero_grad()
            (-crit(m(f_imgs[idx]), f_lbls[idx])).backward()
            sweep_opt.step()
    
    m.eval()
    with torch.no_grad():
        pf = torch.softmax(m(f_imgs), 1).cpu().numpy()
        pr = torch.softmax(m(r_imgs), 1).cpu().numpy()
        preds = m(test_imgs).argmax(1)
        test_acc = (preds == test_lbls).float().mean().item()
    rho, mi_ub = compute_rii_from_probs(pf, pr)
    sr, _ = compute_channel_rank(pf, pr)
    sweep_results.append(dict(epochs=target, rho=float(rho), sigma_ratio=float(sr), test_acc=float(test_acc)))
    print(f"  epochs={target:3d} | ρ={rho:.6f} | σ₂/σ₁={sr:.6f} | acc={test_acc*100:.2f}%")
    if hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()

# ─── Multi-class projection residual ───
print(f"\nMulti-class projection residual...")
heldout_candidates = [c for c in range(100) if c != FORGET_CLASS]
def mean_softmax(indices):
    with torch.no_grad():
        return torch.softmax(m(all_imgs[indices].to(DEVICE)), 1).mean(0).cpu().numpy()

mu_f = mean_softmax(forget_idx)
proj_results = {}
for K in [1, 2, 3, 5, 10, 20]:
    H_rows = [mean_softmax((all_lbls == c).nonzero().squeeze().tolist()) for c in heldout_candidates[:K]]
    H = np.stack(H_rows)
    H_plus = np.linalg.pinv(H)
    residual = mu_f - mu_f @ H_plus @ H
    rho_H = float(np.sum(residual**2) / (np.sum(mu_f**2) + 1e-12))
    proj_results[K] = rho_H
    print(f"  K={K:2d} | ρ_H={rho_H:.6f}")

# ─── Save ───
res = dict(
    epochs=EPOCHS, final_acc=float(acc),
    gradient_sweep=sweep_results,
    projection_residual=proj_results,
    time_sec=time.time()-t0
)
path = os.path.join(OUT_DIR, 'result.json')
with open(path, 'w') as f:
    json.dump(res, f, indent=2, default=str)
print(f"\nSaved to {path}")
print(f"Total time: {time.time()-t0:.0f}s")
