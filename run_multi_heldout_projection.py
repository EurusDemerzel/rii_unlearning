#!/usr/bin/env python3
"""
Multi-class projection residual: replace LOCO with subspace projection.

Idea:
  - Hold out K≥2 classes as "never-seen reference" 
  - Train on remaining C-K classes
  - Forget one trained class f
  - Project μ_f onto row space of H (K held-out class means)
  - ρ_H = ||residual||² / ||μ_f||²  → 0 means "forgotten class looks like unseen data"
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

OUT_DIR = "results/multi_heldout_projection"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 10

# ─── Load MNIST ───
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

all_imgs = train_ds.data.float() / 255.0
all_imgs = (all_imgs - 0.1307) / 0.3081
all_lbls = train_ds.targets

# ─── Config: vary K (number of held-out classes) ───
K_values = [1, 2, 3, 4]
RESULTS = {}

for K in K_values:
    print(f"\n{'='*60}")
    print(f"K = {K} held-out classes")
    print(f"{'='*60}")
    
    # Held-out classes: pick classes 0, 1, ..., K-1 
    heldout_classes = list(range(K))
    # Forget class: pick class 5 (must not be held out)
    forget_class = 5
    # Training classes: all except held-out and forget
    train_classes = [c for c in range(10) if c not in heldout_classes and c != forget_class]
    
    print(f"  Held-out: {heldout_classes}")
    print(f"  Train:    {train_classes}")
    print(f"  Forget:   {forget_class}")
    
    # Indices
    heldout_idx = torch.cat([(all_lbls == c).nonzero().squeeze() for c in heldout_classes]).tolist()
    forget_idx  = (all_lbls == forget_class).nonzero().squeeze().tolist()
    train_idx   = torch.cat([(all_lbls == c).nonzero().squeeze() for c in train_classes]).tolist()
    
    print(f"  n_heldout={len(heldout_idx)}  n_forget={len(forget_idx)}  n_train={len(train_idx)}")
    
    # Train model on training classes only
    train_loader = DataLoader(
        TensorDataset(all_imgs[train_idx].unsqueeze(1), all_lbls[train_idx]),
        batch_size=BATCH_SIZE, shuffle=True)
    
    model = get_model('mnist', DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    
    # Test accuracy
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    cor = tot = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            cor += (model(x).argmax(1) == y).sum().item(); tot += y.size(0)
    test_acc = cor / tot
    print(f"  Test acc: {test_acc*100:.2f}%")
    
    # ─── Compute mean softmax outputs ───
    model.eval()
    def mean_softmax(indices):
        with torch.no_grad():
            x = all_imgs[indices].unsqueeze(1).to(DEVICE)
            return torch.softmax(model(x), 1).mean(0).cpu().numpy()  # (C,)
    
    # Held-out matrix H: K × C
    H_rows = []
    for c in heldout_classes:
        idx = (all_lbls == c).nonzero().squeeze().tolist()
        H_rows.append(mean_softmax(idx))
    H = np.stack(H_rows, axis=0)  # (K, C)
    
    # Forget class mean
    mu_f = mean_softmax(forget_idx)  # (C,)
    
    # Standard RII: forget vs retain set
    retain_idx = train_idx  # all training classes
    mu_r = mean_softmax(retain_idx)
    M_std = np.stack([mu_f, mu_r], axis=0)
    _, S_std, _ = np.linalg.svd(M_std, full_matrices=False)
    rho_std = S_std[1]**2 / (S_std[0]**2 + S_std[1]**2 + 1e-12)
    
    # LOCO-RII: forget vs single held-out (class heldout_classes[0])
    h0_idx = (all_lbls == heldout_classes[0]).nonzero().squeeze().tolist()
    mu_h0 = mean_softmax(h0_idx)
    if mu_h0 is not None:
        M_loco = np.stack([mu_f, mu_h0], axis=0)
        _, S_loco, _ = np.linalg.svd(M_loco, full_matrices=False)
        rho_loco = S_loco[1]**2 / (S_loco[0]**2 + S_loco[1]**2 + 1e-12)
    else:
        rho_loco = None
    
    # ─── NEW: Multi-class projection residual ───
    # Project μ_f onto row space of H
    # H is K×C, compute H⁺ (pseudo-inverse)
    H_plus = np.linalg.pinv(H)  # (C, K)
    mu_f_proj = mu_f @ H_plus @ H  # project: μ_f → row space of H
    residual = mu_f - mu_f_proj
    
    rho_H = np.sum(residual**2) / (np.sum(mu_f**2) + 1e-12)
    
    # Also compute: does projection onto H capture held-out behavior?
    # Check: for each held-out class, how well is it approximated by the projection from OTHER held-outs?
    cross_residuals = []
    for i, c in enumerate(heldout_classes):
        idx = (all_lbls == c).nonzero().squeeze().tolist()
        mu_h = mean_softmax(idx)
        # Project using all OTHER held-out rows (leave-one-out)
        other_idx = [j for j in range(len(heldout_classes)) if j != i]
        if len(other_idx) >= 1:
            H_loo = H[other_idx]  # (K-1, C)
            H_loo_plus = np.linalg.pinv(H_loo)
            mu_h_proj = mu_h @ H_loo_plus @ H_loo
            res = np.sum((mu_h - mu_h_proj)**2) / (np.sum(mu_h**2) + 1e-12)
            cross_residuals.append(res)
    cross_mean = np.mean(cross_residuals) if cross_residuals else 0.0
    
    RESULTS[K] = dict(
        rho_std=float(rho_std),
        rho_loco=float(rho_loco) if rho_loco is not None else None,
        rho_H=float(rho_H),
        heldout_cross_residual_mean=float(cross_mean),
        test_acc=float(test_acc),
    )
    
    print(f"  ρ_std (forget vs retain)         = {rho_std:.6f}")
    if rho_loco is not None:
        print(f"  ρ_loco (forget vs held-out[0])   = {rho_loco:.6f}")
    print(f"  ★ ρ_H  (projection residual)     = {rho_H:.6f}")
    print(f"  Held-out cross residual (mean)   = {cross_mean:.6f}")

# ─── Summary ───
print(f"\n{'='*70}")
print(f"Multi-class Projection Residual Summary")
print(f"{'='*70}")
print(f"{'K':>3} | {'ρ_std':>10} | {'ρ_loco':>10} | {'★ ρ_H':>10} | {'cross_res':>10} | {'acc':>6}")
print(f"{'─'*3} | {'─'*10} | {'─'*10} | {'─'*10} | {'─'*10} | {'─'*6}")
for K, res in sorted(RESULTS.items()):
    loco = f"{res['rho_loco']:.6f}" if res['rho_loco'] is not None else "N/A"
    print(f"{K:3d} | {res['rho_std']:10.6f} | {loco:>10} | {res['rho_H']:10.6f} | {res['heldout_cross_residual_mean']:10.6f} | {res['test_acc']*100:5.1f}%")

path = os.path.join(OUT_DIR, 'result.json')
with open(path, 'w') as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\nSaved to {path}")
