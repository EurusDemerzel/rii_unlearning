#!/usr/bin/env python3
"""
benchmark_cross.py — cross-dataset class-level unlearning benchmark.

Runs the same 7-method × (RII, MHPR, MIA-loss, retain/forget acc) protocol on:
  - CIFAR-10     (7 train classes, forget class 3, held-out {7,8,9})
  - Fashion-MNIST(7 train classes, forget class 3, held-out {7,8,9})
  - CIFAR-100    (20 train classes, forget class 3, held-out {20,21,22})

Outputs results/benchmark_cross/cross_dataset.csv + a printed table.
"""

import os, sys, time, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import SmallCNN

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
BS = 64
EPOCHS = 10          # match benchmark_v2 main table
OUT_DIR = os.path.join("results", "benchmark_cross")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Dataset configs
# ----------------------------------------------------------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
FMNIST_MEAN = (0.2860,)
FMNIST_STD = (0.3530,)

CONFIGS = {
    "cifar10": dict(num_train=7, forget=3, held=[7, 8, 9], ch=3, size=32, nc=10),
    "fashion_mnist": dict(num_train=7, forget=3, held=[7, 8, 9], ch=1, size=28, nc=10),
    "cifar100": dict(num_train=20, forget=3, held=[20, 21, 22], ch=3, size=32, nc=20),
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x.view(x.size(0), -1))))


def load(name, cfg):
    if name == "cifar10":
        t = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        tr = datasets.CIFAR10(root="./data", train=True, download=True, transform=t)
        te = datasets.CIFAR10(root="./data", train=False, download=True, transform=t)
        model_fn = lambda nc: SmallCNN(input_channels=3, num_classes=nc).to(DEVICE)
    elif name == "fashion_mnist":
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(FMNIST_MEAN, FMNIST_STD)])
        tr = datasets.FashionMNIST(root="./data", train=True, download=True, transform=t)
        te = datasets.FashionMNIST(root="./data", train=False, download=True, transform=t)
        model_fn = lambda nc: MLP(28 * 28, 128, nc).to(DEVICE)
    elif name == "cifar100":
        t = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        tr = datasets.CIFAR100(root="./data", train=True, download=True, transform=t)
        te = datasets.CIFAR100(root="./data", train=False, download=True, transform=t)
        model_fn = lambda nc: SmallCNN(input_channels=3, num_classes=nc).to(DEVICE)
    all_imgs = torch.stack([tr[i][0] for i in range(len(tr))])
    all_lbls = torch.tensor(tr.targets)
    test_ldr = torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False)
    return all_imgs, all_lbls, test_ldr, model_fn


def run_dataset(name, cfg):
    print(f"\n{'='*70}\nDataset: {name}\n{'='*70}")
    all_imgs, all_lbls, test_ldr, model_fn = load(name, cfg)
    nc, ch = cfg["nc"], cfg["ch"]

    train_idx = np.concatenate([np.where(all_lbls.numpy() == c)[0] for c in range(cfg["num_train"])])
    forget_cls = cfg["forget"]
    forget_idx = np.where(all_lbls.numpy() == forget_cls)[0]
    retain_idx = np.concatenate([np.where(all_lbls.numpy() == c)[0]
                                 for c in range(cfg["num_train"]) if c != forget_cls])
    held_idx = {c: np.where(all_lbls.numpy() == c)[0] for c in cfg["held"]}

    def ldr(idx, shuffle=True):
        idx = torch.from_numpy(np.asarray(idx))
        ds = torch.utils.data.TensorDataset(all_imgs[idx], all_lbls[idx])
        return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=shuffle)

    forget_loader, retain_loader = ldr(forget_idx), ldr(retain_idx)

    def train(model, loader, epochs, lr=1e-3):
        opt = optim.Adam(model.parameters(), lr=lr); crit = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        return model

    def acc(model, loader):
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(1) == y).sum().item(); total += y.numel()
        return 100.0 * correct / max(total, 1)

    def probs(model, idx):
        idx = torch.from_numpy(np.asarray(idx))
        ds = torch.utils.data.TensorDataset(all_imgs[idx], all_lbls[idx])
        l = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False)
        ps = []
        with torch.no_grad():
            for x, _ in l:
                ps.append(torch.softmax(model(x.to(DEVICE)), 1).cpu().numpy())
        return np.concatenate(ps, 0)

    def losses(model, idx):
        idx = torch.from_numpy(np.asarray(idx))
        ds = torch.utils.data.TensorDataset(all_imgs[idx], all_lbls[idx])
        l = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False)
        out = []
        with torch.no_grad():
            for x, y in l:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out.append(nn.CrossEntropyLoss(reduction="none")(model(x), y).cpu().numpy())
        return np.concatenate(out, 0)

    def rii(pf, pr):
        M = np.stack([pf.mean(0), pr.mean(0)])
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        return S[1] ** 2 / (S[0] ** 2 + S[1] ** 2 + 1e-12)

    def mhpr(pf, phs):
        mu_f = pf.mean(0); H = np.stack([p.mean(0) for p in phs])
        proj = np.linalg.pinv(H) @ H @ mu_f
        return np.sum((mu_f - proj) ** 2) / (np.sum(mu_f ** 2) + 1e-12)

    def mia_auc(lf, lr_):
        from sklearn.metrics import roc_auc_score
        y = np.concatenate([np.ones(len(lf)), np.zeros(len(lr_))])
        s = np.concatenate([-lf, -lr_])
        return roc_auc_score(y, s)

    # base model
    base = model_fn(nc)
    train(base, ldr(train_idx), EPOCHS)
    r_idx = np.random.choice(retain_idx, 5000, replace=False)

    def clone(m):
        m2 = model_fn(nc); m2.load_state_dict(m.state_dict()); return m2

    results = {}
    # NoUnlearn
    m = clone(base); results["NoUnlearn"] = m
    # Retrain
    m = model_fn(nc); train(m, ldr(retain_idx), EPOCHS); results["Retrain"] = m
    # NegGrad
    m = clone(base); crit = nn.CrossEntropyLoss(); opt = optim.Adam(m.parameters(), lr=1e-6); m.train()
    for _ in range(150):
        x, y = next(iter(forget_loader)); x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); (-crit(m(x), y)).backward(); opt.step()
    results["NegGrad"] = m
    # FineTune
    m = clone(base); crit = nn.CrossEntropyLoss()
    opt = optim.Adam(m.parameters(), lr=1e-6); m.train()
    for _ in range(150):
        x, y = next(iter(forget_loader)); x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); (-crit(m(x), y)).backward(); opt.step()
    opt = optim.Adam(m.parameters(), lr=1e-4)
    for _ in range(1):
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    results["FineTune"] = m
    # KED
    m = clone(base); crit = nn.CrossEntropyLoss(); opt = optim.Adam(m.parameters(), lr=1e-4); m.train()
    unif = torch.full((1, nc), 1.0 / nc, device=DEVICE)
    for _ in range(2):
        for x, y in forget_loader:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            nn.functional.kl_div(nn.functional.log_softmax(m(x), 1),
                                 unif.expand(x.size(0), -1), reduction="batchmean").backward(); opt.step()
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    results["KED"] = m
    # BadTeacher
    m = clone(base); crit = nn.CrossEntropyLoss(); opt = optim.Adam(m.parameters(), lr=1e-4); m.train()
    for _ in range(2):
        for x, y in forget_loader:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            crit(m(x), (y + 1) % nc).backward(); opt.step()
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    results["BadTeacher"] = m

    rows = []
    for name_m, m in results.items():
        pf = probs(m, forget_idx)
        pr = probs(m, r_idx)
        phs = [probs(m, held_idx[c][:1000]) for c in cfg["held"]]
        lf = losses(m, forget_idx); lr_ = losses(m, r_idx)
        rows.append({
            "dataset": name, "method": name_m,
            "retain_acc": round(acc(m, retain_loader), 1),
            "forget_acc": round(acc(m, forget_loader), 1),
            "rii": rii(pf, pr), "mhpr": mhpr(pf, phs),
            "mia_loss": mia_auc(lf, lr_),
        })
        print(f"  {name_m:<11} retain={rows[-1]['retain_acc']:>5.1f}% forget={rows[-1]['forget_acc']:>5.1f}% "
              f"RII={rows[-1]['rii']:.2e} MHPR={rows[-1]['mhpr']:.3f} MIA={rows[-1]['mia_loss']:.3f}")
    return rows


def main():
    # CIFAR-100 requires a large slow download; run the two ready datasets by
    # default and allow opt-in with `--all`.
    datasets = ["cifar10", "fashion_mnist"]
    if "--all" in sys.argv:
        datasets = list(CONFIGS.keys())
    all_rows = []
    for name in datasets:
        all_rows.extend(run_dataset(name, CONFIGS[name]))
    with open(os.path.join(OUT_DIR, "cross_dataset.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "retain_acc", "forget_acc", "rii", "mhpr", "mia_loss"])
        w.writeheader(); w.writerows(all_rows)
    print("\nSaved:", os.path.join(OUT_DIR, "cross_dataset.csv"))


if __name__ == "__main__":
    main()
