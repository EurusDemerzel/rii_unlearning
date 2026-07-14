#!/usr/bin/env python3
"""
Experiment 1b: CIFAR-10 class-level forgetting + gradient sweep (multi-class).

Verifies rank-2 → near-rank-1 transition for THREE different classes
to show the effect is class-independent.
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

OUT_DIR = "results/cifar10_class_forget"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 10
FORGET_CLASSES = [3, 1, 5]  # cat, automobile, dog — three different types
GRADIENT_EPOCHS = [0, 1, 3, 10, 30, 100]
GRADIENT_LR = 1e-5
C = 10

# ─── Data ───
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_ds = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_ds  = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

all_imgs = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
all_lbls = torch.tensor(train_ds.targets)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

all_results = {}

for FC in FORGET_CLASSES:
    print(f"\n{'='*60}")
    print(f"Class: {train_ds.classes[FC]} (index {FC})")
    print(f"{'='*60}")

    forget_idx = (all_lbls == FC).nonzero().squeeze().tolist()
    retain_idx = [i for i in range(len(train_ds)) if i not in forget_idx]
    print(f"  Forget: {len(forget_idx)}  Retain: {len(retain_idx)}")

    retain_loader = DataLoader(TensorDataset(all_imgs[retain_idx], all_lbls[retain_idx]),
                                batch_size=BATCH_SIZE, shuffle=True)

    # Train on retain set
    model = get_model('cifar10', DEVICE, model_name='cnn')
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()

    # Test accuracy
    model.eval()
    cor = tot = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            cor += (model(x).argmax(1) == y).sum().item(); tot += y.size(0)
    base_acc = cor / tot
    print(f"  Test acc: {base_acc*100:.2f}%")

    # Gradient sweep (full epochs)
    history = []
    m = get_model('cifar10', DEVICE, model_name='cnn')
    m.load_state_dict(model.state_dict())
    sweep_opt = optim.Adam(m.parameters(), lr=GRADIENT_LR)

    f_imgs = all_imgs[forget_idx].to(DEVICE)
    f_lbls = all_lbls[forget_idx].to(DEVICE)
    r_imgs = all_imgs[retain_idx].to(DEVICE)
    r_lbls = all_lbls[retain_idx].to(DEVICE)
    n_f = len(forget_idx)

    for target_epochs in GRADIENT_EPOCHS:
        # Run gradient ascent for target_epochs
        m.train()
        for _ in range(target_epochs):
            perm = torch.randperm(n_f)
            for i in range(0, n_f, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                sweep_opt.zero_grad()
                (-crit(m(f_imgs[idx]), f_lbls[idx])).backward()
                sweep_opt.step()

        # Compute metrics
        m.eval()
        with torch.no_grad():
            pf = torch.softmax(m(f_imgs), 1).cpu().numpy()
            pr = torch.softmax(m(r_imgs), 1).cpu().numpy()
        rho, mi_ub = compute_rii_from_probs(pf, pr)
        sr, _ = compute_channel_rank(pf, pr)

        with torch.no_grad():
            cor = tot = 0
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                cor += (m(x).argmax(1) == y).sum().item(); tot += y.size(0)
        test_acc = cor / tot

        history.append(dict(steps=target_epochs, rho=float(rho), sigma_ratio=float(sr),
                            mi_ub=float(mi_ub), test_acc=float(test_acc)))
        print(f"  epochs={target_epochs:3d} | ρ={rho:.6f} | σ₂/σ₁={sr:.6f} | acc={test_acc*100:.2f}%")

    all_results[train_ds.classes[FC]] = history

# ─── Summary ───
print(f"\n{'='*70}")
print(f"CIFAR-10 Multi-Class Gradient Sweep Summary")
print(f"{'='*70}")
for cls_name, hist in all_results.items():
    rho0 = hist[0]['rho']
    # find final non-zero epoch
    final = [h for h in hist if h['steps'] > 0]
    rhoN = final[-1]['rho'] if final else rho0
    ratio = rho0 / max(rhoN, 1e-10)
    print(f"  {cls_name:>12s}: ρ_before={rho0:.4f}  ρ_after={rhoN:.6f}  ↓{ratio:.0f}×")

# Save
path = os.path.join(OUT_DIR, 'result_multi_class.json')
with open(path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved to {path}")
