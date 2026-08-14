#!/usr/bin/env python3
"""
benchmark_v2.py — CIFAR-10 class-level unlearning benchmark + full evaluation.

Implements Phase 1 (real unlearning benchmark) and Phase 2 (same-benchmark
comparison with existing verification methods) from the APIN revision plan.

Protocol (class-level, K=3 held-out):
  - Train classes      : {0,1,2,3,4,5,6}   (7 classes, incl. forget class)
  - Forget class       : 3 (cat)
  - Retain classes     : {0,1,2,4,5,6}     (6 classes)
  - Held-out classes   : {7,8,9}           (unseen, MHPR reference, K=3)

Unlearning methods:
  NoUnlearn / Retrain (oracle) / NegGrad / FineTune / KED / BadTeacher / SISA

Evaluation metrics (same benchmark, all methods):
  retain_acc, forget_acc, RII (rho), MHPR (K=3),
  MIA-loss AUC, MIA-conf AUC,               (MIA baselines)
  posterior-diff (TAPE-style),              (posterior difference)
  repr-MMD      (RULER-style),              (representation MMD)
  residual-probe AUC (RUB-style)            (residual knowledge probe)

Outputs: results/benchmark_v2/results.csv + fig1_methods.png + fig2_corr.png
"""

import os, sys, json, time, csv, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model, clone_model


def load_data(data_root="./data"):
    """Load CIFAR-10 or Fashion-MNIST with plain normalization."""
    if DS_NAME == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train_ds = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    all_imgs = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    all_lbls = torch.tensor(train_ds.targets)
    test_ldr = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
    return all_imgs, all_lbls, len(train_ds), test_ldr, test_ds

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
if "--seed" in sys.argv:
    SEED = int(sys.argv[sys.argv.index("--seed") + 1])
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

DS_NAME, MODEL_NAME = "cifar10", "cnn"
if "--fashion_mnist" in sys.argv:
    DS_NAME, MODEL_NAME = "fashion_mnist", "mlp"
FORGET_CLASS = 3
HELD_OUT_CLASSES = [7, 8, 9]           # K=3 held-out unseen classes (MHPR)
TRAIN_CLASSES = [0, 1, 2, 3, 4, 5, 6]  # 7 classes incl. forget class
RETAIN_CLASSES = [c for c in TRAIN_CLASSES if c != FORGET_CLASS]

EPOCHS = 10
LR = 1e-3
BS = 64
OUT_DIR = os.path.join("results", f"benchmark_v2_{DS_NAME}_s{SEED}")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
print(f"Device: {DEVICE} | Dataset: {DS_NAME} | Model: {MODEL_NAME}")
all_imgs, all_lbls, N_total, test_ldr, _ = load_data()

def idx_of_class(c):
    return torch.where(all_lbls == c)[0].numpy()

train_idx = np.concatenate([idx_of_class(c) for c in TRAIN_CLASSES])
forget_idx = idx_of_class(FORGET_CLASS)
retain_idx = np.concatenate([idx_of_class(c) for c in RETAIN_CLASSES])
heldout_idx = {c: idx_of_class(c) for c in HELD_OUT_CLASSES}

# Build a forget/retain test loader for accuracy on the forget class
def build_class_loader(class_ids, shuffle=False, n_max=None):
    idx = np.concatenate([idx_of_class(c) for c in class_ids])
    if n_max is not None:
        idx = np.random.choice(idx, min(n_max, len(idx)), replace=False)
    imgs = all_imgs[torch.from_numpy(idx)]
    lbls = all_lbls[torch.from_numpy(idx)]
    ds = torch.utils.data.TensorDataset(imgs, lbls)
    return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=shuffle)

train_loader   = build_class_loader(TRAIN_CLASSES, shuffle=True)
retain_loader  = build_class_loader(RETAIN_CLASSES, shuffle=True)
forget_loader  = build_class_loader([FORGET_CLASS], shuffle=True)
heldout_loader = {c: build_class_loader([c]) for c in HELD_OUT_CLASSES}

# ----------------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------------
def train(model, loader, epochs, lr=LR, device=DEVICE):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    return model

def evaluate_acc(model, loader, device=DEVICE):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct / max(total, 1)

# ----------------------------------------------------------------------------
# Unlearning methods
# ----------------------------------------------------------------------------
def method_none(model):
    return clone_model(model, DS_NAME, DEVICE, MODEL_NAME), 0.0

def method_retrain(model):
    t0 = time.time()
    m = get_model(DS_NAME, DEVICE, MODEL_NAME)
    train(m, retain_loader, EPOCHS)
    return m, time.time() - t0

def method_neggrad(model, steps=150, lr=1e-6):
    m = clone_model(model, DS_NAME, DEVICE, MODEL_NAME)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(m.parameters(), lr=lr)
    m.train()
    t0 = time.time()
    for _ in range(steps):
        x, y = next(iter(forget_loader))
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); (-crit(m(x), y)).backward(); opt.step()
    return m, time.time() - t0

def method_finetune(model, ascent=150, a_lr=1e-6, retain_epochs=1, r_lr=1e-4):
    m = clone_model(model, DS_NAME, DEVICE, MODEL_NAME)
    crit = nn.CrossEntropyLoss()
    m.train(); t0 = time.time()
    opt = optim.Adam(m.parameters(), lr=a_lr)
    for _ in range(ascent):
        x, y = next(iter(forget_loader))
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); (-crit(m(x), y)).backward(); opt.step()
    opt = optim.Adam(m.parameters(), lr=r_lr)
    for _ in range(retain_epochs):
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    return m, time.time() - t0

def method_ked(model, epochs=2, lr=1e-4):
    """Knowledge erosion: minimize KL(f(x)||uniform) on forget + CE on retain."""
    m = clone_model(model, DS_NAME, DEVICE, MODEL_NAME)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(m.parameters(), lr=lr)
    m.train(); t0 = time.time()
    C = 10
    uniform = torch.full((1, C), 1.0 / C, device=DEVICE)
    for _ in range(epochs):
        for x, y in forget_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = m(x)
            kl = nn.functional.kl_div(nn.functional.log_softmax(logits, 1),
                                      uniform.expand(x.size(0), -1), reduction="batchmean")
            kl.backward(); opt.step()
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    return m, time.time() - t0

def method_bad_teacher(model, epochs=2, lr=1e-4):
    """Bad teacher: train on forget set with wrong labels + CE on retain."""
    m = clone_model(model, DS_NAME, DEVICE, MODEL_NAME)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(m.parameters(), lr=lr)
    m.train(); t0 = time.time()
    # wrong labels: shift forget labels by +1 (mod C), never == true label
    for _ in range(epochs):
        for x, y in forget_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            wrong = (y + 1) % 10
            opt.zero_grad(); crit(m(x), wrong).backward(); opt.step()
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    return m, time.time() - t0

def method_sisa(model, S=5, T=2, eps=1):
    """SISA: train S shards; on unlearning, retrain affected shards w/o forget."""
    t0 = time.time()
    perm = np.random.permutation(len(train_idx))
    shards = np.array_split(train_idx[perm], S)
    shard_models = []
    for s in range(S):
        sh = shards[s]
        m = get_model(DS_NAME, DEVICE, MODEL_NAME)
        sl = np.array_split(sh, T)
        for t in range(T):
            m = train(m, _from_idx(sl[t]), eps)
        shard_models.append(m)
    # unlearn: retrain affected shards (those containing forget-class samples)
    unlearned = []
    for s in range(S):
        sh = shards[s]
        if FORGET_CLASS in all_lbls[torch.from_numpy(sh)].numpy():
            clean = sh[all_lbls[torch.from_numpy(sh)].numpy() != FORGET_CLASS]
            m = get_model(DS_NAME, DEVICE, MODEL_NAME)
            if len(clean):
                m = train(m, _from_idx(clean), eps * T)
            unlearned.append(m)
        else:
            unlearned.append(clone_model(shard_models[s], DS_NAME, DEVICE, MODEL_NAME))
    return unlearned, time.time() - t0

def _from_idx(idx):
    idx = torch.from_numpy(np.asarray(idx))
    ds = torch.utils.data.TensorDataset(all_imgs[idx], all_lbls[idx])
    return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=True)

# SISA aggregation evaluation
def sisa_probs(model, x):
    p = None
    for m in model:
        m.eval()
        p = torch.softmax(m(x), 1) if p is None else p + torch.softmax(m(x), 1)
    return p / len(model)

def sisa_predict(model, loader):
    model = model  # list
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (sisa_probs(model, x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def get_probs(model, idx, is_sisa=False):
    idx = torch.from_numpy(np.asarray(idx))
    ds = torch.utils.data.TensorDataset(all_imgs[idx], all_lbls[idx])
    ldr = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False)
    ps = []
    with torch.no_grad():
        for x, _ in ldr:
            x = x.to(DEVICE)
            ps.append(sisa_probs(model, x).cpu().numpy() if is_sisa
                      else torch.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(ps, 0)

def get_logits_loss(model, idx, is_sisa=False):
    idx = torch.from_numpy(np.asarray(idx))
    ds = torch.utils.data.TensorDataset(all_imgs[idx], all_lbls[idx])
    ldr = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False)
    logits, losses = [], []
    crit = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for x, y in ldr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            l = sisa_probs(model, x) if is_sisa else model(x)
            logits.append(l.cpu().numpy())
            losses.append(crit(l, y).cpu().numpy())
    return np.concatenate(logits, 0), np.concatenate(losses, 0)

def get_features(model, idx, is_sisa=False):
    """Penultimate (fc1) features via forward hook."""
    idx = torch.from_numpy(np.asarray(idx))
    ds = torch.utils.data.TensorDataset(all_imgs[idx], all_lbls[idx])
    ldr = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False)
    feats = []
    def hook_fn(m, i, o):
        feats.append(o.detach().cpu().numpy())
    handle = None
    if is_sisa:
        # use first shard's hook; SISA features approximated by shard 0
        handle = model[0].fc1.register_forward_hook(hook_fn)
    else:
        handle = model.fc1.register_forward_hook(hook_fn)
    with torch.no_grad():
        for x, _ in ldr:
            x = x.to(DEVICE)
            if is_sisa:
                model[0](x)
            else:
                model(x)
    handle.remove()
    return np.concatenate(feats, 0)

def compute_rii(probs_f, probs_r):
    M = np.stack([probs_f.mean(0), probs_r.mean(0)])
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    rho = S[1]**2 / (S[0]**2 + S[1]**2 + 1e-12)
    return rho, S[1] / max(S[0], 1e-12)

def compute_mhpr(probs_f, probs_h_list):
    """MHPR: project forget mean onto span of held-out means."""
    mu_f = probs_f.mean(0)
    H = np.stack([ph.mean(0) for ph in probs_h_list])          # (K, C)
    Hp = np.linalg.pinv(H)
    proj = Hp @ H @ mu_f
    return np.sum((mu_f - proj)**2) / (np.sum(mu_f**2) + 1e-12)

def auc(scores_pos, scores_neg):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    s = np.concatenate([scores_pos, scores_neg])
    return roc_auc_score(y, s)

def rbf_mmd(X, Y, sigma=None):
    """Unbiased RBF MMD estimator on subsamples; median-heuristic sigma."""
    n = min(800, len(X), len(Y))
    Xs, Ys = X[:n].astype(np.float64), Y[:n].astype(np.float64)
    if sigma is None:
        Z = np.concatenate([Xs, Ys])
        s = np.random.choice(len(Z), min(200, len(Z)), replace=False)
        d2 = ((Z[s][:, None] - Z[s][None, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0])
        sigma = float(np.sqrt(med / 2.0)) if med > 0 else 1.0
    def k(a, b):
        d2 = ((a[:, None] - b[None, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * sigma ** 2))
    return k(Xs, Xs).mean() + k(Ys, Ys).mean() - 2 * k(Xs, Ys).mean()

def residual_probe_auc(feat_f, feat_h, test_frac=0.3):
    """Linear probe: distinguish forget features from held-out features."""
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([feat_f, feat_h])
    y = np.concatenate([np.ones(len(feat_f)), np.zeros(len(feat_h))])
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    n_test = int(test_frac * len(X))
    clf = LogisticRegression(max_iter=500)
    clf.fit(X[:-n_test], y[:-n_test])
    s = clf.predict_proba(X[-n_test:])[:, 1]
    return _auc_from_probs(s, y[-n_test:])

def _auc_from_probs(s, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)

# ----------------------------------------------------------------------------
# Full evaluation of one unlearned model
# ----------------------------------------------------------------------------
def evaluate(model, is_sisa=False, orig=None):
    d = {}
    # accuracy
    d["retain_acc"] = (sisa_predict if is_sisa else evaluate_acc)(model, retain_loader)
    d["forget_acc"] = (sisa_predict if is_sisa else evaluate_acc)(model, forget_loader)
    # reference subsets (uniform across retain classes)
    r_idx = np.random.choice(retain_idx, 5000, replace=False)
    # RII + MHPR
    pf = get_probs(model, forget_idx, is_sisa)
    pr = get_probs(model, r_idx, is_sisa)
    d["rii_rho"], d["sigma_ratio"] = compute_rii(pf, pr)
    ph = [get_probs(model, heldout_idx[c][:1000], is_sisa) for c in HELD_OUT_CLASSES]
    d["mhpr"] = compute_mhpr(pf, ph)
    # MIA baselines (loss & confidence) — positive = forget member
    _, lf = get_logits_loss(model, forget_idx, is_sisa)
    _, lr_ = get_logits_loss(model, r_idx, is_sisa)
    d["mia_loss_auc"] = auc(-lf, -lr_)            # higher score = lower loss = member
    conf_f = get_probs(model, forget_idx, is_sisa).max(1)
    conf_r = get_probs(model, r_idx, is_sisa).max(1)
    d["mia_conf_auc"] = auc(conf_f, conf_r)
    # TAPE-style: posterior difference vs original (forget set)
    if orig is not None:
        po = get_probs(orig, forget_idx, is_sisa=False)   # orig is never SISA
        d["posterior_diff"] = float(np.mean(np.linalg.norm(pf - po, axis=1)))
    else:
        d["posterior_diff"] = np.nan
    # RULER-style: representation MMD (forget vs retain features)
    ff = get_features(model, forget_idx, is_sisa)
    fr = get_features(model, r_idx, is_sisa)
    d["repr_mmd"] = rbf_mmd(ff, fr)
    # representation anomaly: forget features vs held-out (unseen) features
    fh_feat = get_features(model, np.concatenate([heldout_idx[c][:1000] for c in HELD_OUT_CLASSES]), is_sisa)
    d["repr_mmd_holdout"] = rbf_mmd(ff, fh_feat)
    # RUB-style: residual probe AUC (forget vs held-out features)
    d["residual_probe_auc"] = residual_probe_auc(ff, fh_feat)
    return d

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    results = []
    t_all = time.time()

    # --- base model ---
    print("\n[1/8] Training base model on 7 classes ...")
    base = get_model(DS_NAME, DEVICE, MODEL_NAME)
    train(base, train_loader, EPOCHS)
    print(f"      base retain_acc={evaluate_acc(base, retain_loader):.1f}% "
          f"forget_acc={evaluate_acc(base, forget_loader):.1f}%")

    methods = [
        ("NoUnlearn", lambda m: method_none(m)),
        ("Retrain",   lambda m: method_retrain(m)),
        ("NegGrad",   lambda m: method_neggrad(m)),
        ("FineTune",  lambda m: method_finetune(m)),
        ("KED",       lambda m: method_ked(m)),
        ("BadTeacher",lambda m: method_bad_teacher(m)),
        ("SISA",      lambda m: method_sisa(m)),
    ]

    saved_models = []
    for name, fn in methods:
        print(f"[*] Running {name} ...")
        m, t = fn(base)
        is_sisa = isinstance(m, list)
        ev = evaluate(m, is_sisa=is_sisa, orig=base)
        ev["method"] = name
        ev["time_s"] = round(t, 2)
        ev["is_sisa"] = is_sisa
        results.append(ev)
        saved_models.append((name, m))
        print(f"    retain={ev['retain_acc']:.1f}% forget={ev['forget_acc']:.1f}% "
              f"RII={ev['rii_rho']:.2e} MHPR={ev['mhpr']:.3f} "
              f"MIA-loss={ev['mia_loss_auc']:.3f} MIA-conf={ev['mia_conf_auc']:.3f}")

    # --- save CSV ---
    os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)
    for name, mm in saved_models:
        if isinstance(mm, list):
            for i, sh in enumerate(mm):
                torch.save(sh.state_dict(), os.path.join(OUT_DIR, "models", f"{name}_sh{i}.pt"))
        else:
            torch.save(mm.state_dict(), os.path.join(OUT_DIR, "models", f"{name}.pt"))
    print("Models saved to", os.path.join(OUT_DIR, "models"))
    cols = ["method", "retain_acc", "forget_acc", "rii_rho", "sigma_ratio", "mhpr",
            "mia_loss_auc", "mia_conf_auc", "posterior_diff", "repr_mmd",
            "repr_mmd_holdout", "residual_probe_auc", "time_s"]
    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c, np.nan) for c in cols})
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in results[-1].items()} , f, indent=2, default=str)

    # --- print report ---
    print("\n" + "=" * 100)
    print("FULL EVALUATION REPORT (CIFAR-10 class-level, forget=cat, K=3 held-out)")
    print("=" * 100)
    hdr = f"{'method':<12}{'retain':>8}{'forget':>8}{'RII':>10}{'MHPR':>8}{'MIA-loss':>9}{'MIA-conf':>9}{'postDiff':>9}{'reprMMD':>9}{'mmdHO':>9}{'probeAUC':>9}{'time':>7}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['method']:<12}{r['retain_acc']:>7.1f}%{r['forget_acc']:>7.1f}%"
              f"{r['rii_rho']:>10.2e}{r['mhpr']:>8.4f}"
              f"{r['mia_loss_auc']:>9.3f}{r['mia_conf_auc']:>9.3f}"
              f"{r['posterior_diff']:>9.4f}{r['repr_mmd']:>9.4f}"
              f"{r['repr_mmd_holdout']:>9.4f}{r['residual_probe_auc']:>9.3f}{r['time_s']:>7.1f}")
    print("-" * len(hdr))
    print(f"Total wall time: {time.time()-t_all:.1f}s")

    # --- figures ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["method"] for r in results]
        rii = [r["rii_rho"] for r in results]
        mhpr = [r["mhpr"] for r in results]
        mia_l = [r["mia_loss_auc"] for r in results]

        # Fig 1: RII & MHPR (log) + MIA-loss
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        x = np.arange(len(names))
        ax[0].bar(x - 0.2, np.log10(np.maximum(rii, 1e-8)), 0.4, label="log10 RII")
        ax[0].bar(x + 0.2, np.log10(np.maximum(mhpr, 1e-8)), 0.4, label="log10 MHPR")
        ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=20)
        ax[0].set_ylabel("log10 metric"); ax[0].set_title("RII & MHPR by method")
        ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].bar(x, mia_l, 0.5, color="coral", label="MIA-loss AUC")
        ax[1].axhline(0.5, ls="--", color="k", lw=1)
        ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=20)
        ax[1].set_ylabel("AUC"); ax[1].set_title("MIA-loss AUC (lower=better)")
        ax[1].legend(); ax[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "fig1_methods.png"), dpi=150)
        plt.close()

        # Fig 2: correlation heatmap of all metrics
        keys = ["rii_rho", "mhpr", "mia_loss_auc", "mia_conf_auc",
                "posterior_diff", "repr_mmd", "repr_mmd_holdout",
                "residual_probe_auc", "forget_acc", "retain_acc"]
        M = np.array([[r[k] for k in keys] for r in results])
        C = np.corrcoef(M.T)
        fig, ax = plt.subplots(figsize=(8.5, 7))
        im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys)
        for i in range(len(keys)):
            for j in range(len(keys)):
                ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("Metric correlation across unlearning methods")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "fig2_corr.png"), dpi=150)
        plt.close()
        print("\nFigures saved to", OUT_DIR)
    except Exception as e:
        print("Figure generation skipped:", e)

if __name__ == "__main__":
    main()
