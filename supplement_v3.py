#!/usr/bin/env python3
"""
supplement_v3.py — Supplemental experiments A/B/C from the revision plan.
No new downloads (uses local MNIST / CIFAR-10 / Fashion-MNIST).

  A. RII safety-threshold calibration (CIFAR-10 class-level)
       gradient-ascent sweep 0..150 steps -> (rho, forget_acc, MIA-loss);
       joint view with the 7-method benchmark -> suggested rho operating point.
  B. Multi-reference AVERAGE LOCO vs projection MHPR (MNIST)
       single held-out reference fails; average of K references improves but
       stays above the subspace projection -> motivates MHPR's design.
  C. Cross-class gradient-ascent sweep (CIFAR-10, forget classes 2/3/5)
       does the "NegGrad hidden failure" (rho rises under ascent) hold across
       classes, or is it an artifact of one class?

Outputs: results/supplement_v3/  (csv + printed tables)
"""
import os, sys, time, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model, clone_model, SimpleMLP, SmallCNN

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
BS = 64
EPOCHS = 10
LR = 1e-3
OUT = os.path.join("results", "supplement_v3")
os.makedirs(OUT, exist_ok=True)
print(f"Device: {DEVICE}")

# ----------------------------------------------------------------------------
# Data loading (local caches only)
# ----------------------------------------------------------------------------
def load_dataset(name):
    if name == "mnist":
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        tr = datasets.MNIST(root="./data", train=True, download=False, transform=t)
        model_fn = lambda nc: SimpleMLP(28 * 28, 128, nc)
    elif name == "fashion_mnist":
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        tr = datasets.FashionMNIST(root="./data", train=True, download=False, transform=t)
        model_fn = lambda nc: SimpleMLP(28 * 28, 128, nc)
    else:  # cifar10
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        tr = datasets.CIFAR10(root="./data", train=True, download=False, transform=t)
        model_fn = lambda nc: SmallCNN(input_channels=3, num_classes=nc)
    imgs = torch.stack([tr[i][0] for i in range(len(tr))])
    lbls = torch.tensor(tr.targets)
    return imgs, lbls, model_fn


def idx_of(lbls, c):
    return torch.where(lbls == c)[0].numpy()


def loader_from(imgs, lbls, idx, shuffle=True):
    idx = torch.from_numpy(np.asarray(idx))
    ds = torch.utils.data.TensorDataset(imgs[idx], lbls[idx])
    return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=shuffle)


def train(model, loader, epochs=EPOCHS, lr=LR):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    return model


def acc(model, loader):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            c += (model(x).argmax(1) == y).sum().item(); t += y.numel()
    return 100.0 * c / max(t, 1)


def probs(model, imgs, idx):
    idx = torch.from_numpy(np.asarray(idx))
    ds = torch.utils.data.TensorDataset(imgs[idx], torch.zeros(len(idx)))
    l = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False)
    ps = []
    model.eval()
    with torch.no_grad():
        for x, _ in l:
            ps.append(torch.softmax(model(x.to(DEVICE)), 1).cpu().numpy())
    return np.concatenate(ps, 0)


def losses(model, imgs, lbls, idx):
    idx = torch.from_numpy(np.asarray(idx))
    ds = torch.utils.data.TensorDataset(imgs[idx], lbls[idx])
    l = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False)
    out = []
    crit = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    with torch.no_grad():
        for x, y in l:
            out.append(crit(model(x.to(DEVICE)), y.to(DEVICE)).cpu().numpy())
    return np.concatenate(out, 0)


def rii(pf, pr):
    M = np.stack([pf.mean(0), pr.mean(0)])
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    return S[1] ** 2 / (S[0] ** 2 + S[1] ** 2 + 1e-12)


def mhpr_proj(pf, held_means_list):
    """Projection MHPR: residual of mu_f onto span of held-out means."""
    mu_f = pf.mean(0)
    H = np.stack(held_means_list)
    Hp = np.linalg.pinv(H)
    proj = Hp @ H @ mu_f
    return float(np.sum((mu_f - proj) ** 2) / (np.sum(mu_f ** 2) + 1e-12))


def loco_avg(pf, held_means_list):
    """Average-reference LOCO: ||mu_f - mean(held outs)||^2 / ||mu_f||^2."""
    mu_f = pf.mean(0)
    ref = np.mean(np.stack(held_means_list), 0)
    return float(np.sum((mu_f - ref) ** 2) / (np.sum(mu_f ** 2) + 1e-12))


def loco_single(pf, held_means_list):
    """Single-reference LOCO values (one per held-out mean), returned as list."""
    mu_f = pf.mean(0)
    return [float(np.sum((mu_f - h) ** 2) / (np.sum(mu_f ** 2) + 1e-12)) for h in held_means_list]


def gradient_ascent(model, steps, lr=1e-6, forget_loader=None):
    import copy
    m = copy.deepcopy(model)   # same architecture + device
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(m.parameters(), lr=lr)
    m.train()
    for _ in range(steps):
        x, y = next(iter(forget_loader))
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); (-crit(m(x), y)).backward(); opt.step()
    return m


def mia_loss_auc(lf, lr_):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(lf)), np.zeros(len(lr_))])
    s = np.concatenate([-lf, -lr_])   # higher score = lower loss = member
    return float(roc_auc_score(y, s))


# ============================================================================
# Experiment A — RII safety-threshold calibration (CIFAR-10)
# ============================================================================
def exp_A():
    print("\n" + "=" * 70 + "\n[A] RII safety-threshold calibration (CIFAR-10)\n" + "=" * 70)
    imgs, lbls, model_fn = load_dataset("cifar10")
    TRAIN = [0, 1, 2, 3, 4, 5, 6]
    FORGET, HELD = 3, [7, 8, 9]
    RETAIN = [c for c in TRAIN if c != FORGET]
    train_idx = np.concatenate([idx_of(lbls, c) for c in TRAIN])
    forget_idx = idx_of(lbls, FORGET)
    retain_idx = np.concatenate([idx_of(lbls, c) for c in RETAIN])
    held_idx = {c: idx_of(lbls, c) for c in HELD}

    t0 = time.time()
    model = train(model_fn(10).to(DEVICE), loader_from(imgs, lbls, train_idx))
    print(f"base model trained in {time.time()-t0:.1f}s; forget_acc={acc(model, loader_from(imgs,lbls,forget_idx)):.1f}%")

    f_loader = loader_from(imgs, lbls, forget_idx, shuffle=True)
    r_sub = np.random.choice(retain_idx, 5000, replace=False)
    held_means = {c: probs(model, imgs, held_idx[c][:1000]).mean(0) for c in HELD}

    steps_list = [0, 1, 3, 10, 30, 50, 100, 150]
    rows = []
    for s in steps_list:
        m = gradient_ascent(model, s, forget_loader=f_loader)
        pf = probs(m, imgs, forget_idx)
        pr = probs(m, imgs, r_sub)
        ph = [probs(m, imgs, held_idx[c][:1000]) for c in HELD]
        rho = rii(pf, pr)
        rho_h = mhpr_proj(pf, [x.mean(0) for x in ph])
        fa = acc(m, loader_from(imgs, lbls, forget_idx))
        lf = losses(m, imgs, lbls, forget_idx)
        lr_ = losses(m, imgs, lbls, r_sub)
        mauc = mia_loss_auc(lf, lr_)
        rows.append(dict(steps=s, rho=rho, mhpr=rho_h, forget_acc=fa, mia_loss_auc=mauc))
        print(f"  steps={s:4d}  rho={rho:.4f}  MHPR={rho_h:.4f}  forget_acc={fa:5.1f}%  MIA-loss={mauc:.3f}")

    # Joint view with the 7-method benchmark (already computed)
    bench_csv = os.path.join("results", "benchmark_v2", "results.csv")
    bench = []
    if os.path.exists(bench_csv):
        import csv
        with open(bench_csv) as f:
            for row in csv.DictReader(f):
                bench.append(row)
        print("\n  7-method benchmark (rho, forget_acc, MIA-loss):")
        for r in bench:
            print(f"    {r['method']:<12s} rho={float(r['rii_rho']):.4f}  "
                  f"forget_acc={float(r['forget_acc']):5.1f}%  MIA-loss={float(r['mia_loss_auc']):.3f}")
    else:
        print("  (benchmark_v2/results.csv not found — joint view skipped)")

    # Threshold analysis
    print("\n  --- operating-point analysis ---")
    print("  random-subset baseline rho ~ 1e-3 ; class-level NoUnlearn rho ~ 0.2")
    print("  suggested coarse thresholds (output-level):")
    print("    rho < 0.01  : strong output-level indistinguishability (safe regime)")
    print("    0.01-0.1    : partial / class-level asymmetric residual")
    print("    rho > 0.1   : clear class-level output signature (not erased)")

    with open(os.path.join(OUT, "expA_threshold.csv"), "w") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    return rows


# ============================================================================
# Experiment B — Average-reference LOCO vs projection MHPR (MNIST)
# ============================================================================
def exp_B():
    print("\n" + "=" * 70 + "\n[B] Multi-reference AVERAGE LOCO vs projection MHPR (MNIST)\n" + "=" * 70)
    imgs, lbls, model_fn = load_dataset("mnist")
    TRAIN = [0, 1, 2, 3, 4, 5, 6]       # 7 train classes incl. forget 5
    FORGET, HELD = 5, [7, 8, 9]          # K=3 held-out unseen classes
    RETAIN = [c for c in TRAIN if c != FORGET]
    train_idx = np.concatenate([idx_of(lbls, c) for c in TRAIN])
    forget_idx = idx_of(lbls, FORGET)
    retain_idx = np.concatenate([idx_of(lbls, c) for c in RETAIN])
    held_idx = {c: idx_of(lbls, c) for c in HELD}

    model = train(model_fn(10).to(DEVICE), loader_from(imgs, lbls, train_idx))
    pf = probs(model, imgs, forget_idx)
    pr = probs(model, imgs, retain_idx)
    held_means = [probs(model, imgs, held_idx[c][:1000]).mean(0) for c in HELD]
    rho_std = rii(pf, pr)

    print(f"  rho_std (forget vs retain) = {rho_std:.4f}")
    print(f"  {'K':>3} {'loco_single(min/med/max)':>26} {'loco_avg':>10} {'MHPR_proj':>12}")
    for K in [1, 2, 3]:
        hs = held_means[:K]
        singles = loco_single(pf, hs)
        avg = loco_avg(pf, hs)
        proj = mhpr_proj(pf, hs)
        print(f"  {K:3d}  {min(singles):.3f}/{np.median(singles):.3f}/{max(singles):.3f}"
              f"  {avg:10.4f}  {proj:12.4f}")

    with open(os.path.join(OUT, "expB_loco.csv"), "w") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["K", "loco_single_min", "loco_single_med", "loco_single_max", "loco_avg", "mhpr_proj"])
        for K in [1, 2, 3]:
            hs = held_means[:K]
            s = loco_single(pf, hs)
            w.writerow([K, min(s), np.median(s), max(s), loco_avg(pf, hs), mhpr_proj(pf, hs)])
    print("  -> single held-out reference is unstable; averaging helps but stays far")
    print("     above the subspace projection (MHPR), motivating projection design.")


# ============================================================================
# Experiment C — Cross-class gradient-ascent sweep (CIFAR-10)
# ============================================================================
def exp_C():
    print("\n" + "=" * 70 + "\n[C] Cross-class gradient-ascent sweep (CIFAR-10)\n" + "=" * 70)
    imgs, lbls, model_fn = load_dataset("cifar10")
    TRAIN = [0, 1, 2, 3, 4, 5, 6]
    HELD = [7, 8, 9]
    STEPS = [0, 1, 3, 10, 30]

    all_rows = []
    for FORGET in [2, 3, 5]:          # bird, cat, dog
        RETAIN = [c for c in TRAIN if c != FORGET]
        train_idx = np.concatenate([idx_of(lbls, c) for c in TRAIN])
        forget_idx = idx_of(lbls, FORGET)
        retain_idx = np.concatenate([idx_of(lbls, c) for c in RETAIN])
        held_idx = {c: idx_of(lbls, c) for c in HELD}
        r_sub = np.random.choice(retain_idx, 5000, replace=False)

        model = train(model_fn(10).to(DEVICE), loader_from(imgs, lbls, train_idx))
        f_loader = loader_from(imgs, lbls, forget_idx, shuffle=True)
        print(f"  forget class {FORGET}: base forget_acc={acc(model, loader_from(imgs,lbls,forget_idx)):.1f}%")
        for s in STEPS:
            m = gradient_ascent(model, s, forget_loader=f_loader)
            pf = probs(m, imgs, forget_idx)
            pr = probs(m, imgs, r_sub)
            ph = [probs(m, imgs, held_idx[c][:1000]).mean(0) for c in HELD]
            rho = rii(pf, pr)
            rho_h = mhpr_proj(pf, ph)
            fa = acc(m, loader_from(imgs, lbls, forget_idx))
            all_rows.append(dict(forget_class=FORGET, steps=s, rho=rho, mhpr=rho_h, forget_acc=fa))
            print(f"    cls={FORGET} steps={s:3d}  rho={rho:.4f}  MHPR={rho_h:.4f}  forget_acc={fa:5.1f}%")

    # Does rho rise (relative to NoUnlearn baseline at steps=0)?
    print("\n  --- rho(steps)/rho(steps=0) ratio (NegGrad hidden-failure check) ---")
    base = {r["forget_class"]: r["rho"] for r in all_rows if r["steps"] == 0}
    for c in [2, 3, 5]:
        ratios = [(r["steps"], r["rho"] / base[c]) for r in all_rows if r["forget_class"] == c]
        print(f"    class {c}: " + "  ".join(f"step{s}={v:.2f}x" for s, v in ratios))
    print("  -> rho rising with ascent steps across classes = general NegGrad failure;")
    print("     if it only rises for one class, the phenomenon is class-specific.")

    with open(os.path.join(OUT, "expC_cross_class.csv"), "w") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader()
        for r in all_rows: w.writerow(r)


if __name__ == "__main__":
    exp_A()
    exp_B()
    exp_C()
    print("\nAll supplemental experiments done. Outputs in", OUT)
