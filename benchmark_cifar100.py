#!/usr/bin/env python3
"""
benchmark_cifar100.py — CIFAR-100 class-level unlearning benchmark (plan item #1).

Data: local HuggingFace parquet at ~/cifar100/cifar100/ (no network needed).
Protocol (mirrors benchmark_cross cifar100 config):
  - Train classes  : 0..19 (20 classes, incl. forget class)
  - Forget class   : 3
  - Retain classes : all train classes except 3
  - Held-out (MHPR): {20,21,22}  (K=3 unseen)
  - Model          : SmallCNN(num_classes=20), 10 epochs, no augmentation
  - Methods        : NoUnlearn / Retrain / NegGrad / FineTune / KED / BadTeacher
  - Metrics        : retain_acc, forget_acc, RII, MHPR(K=3), MIA-loss AUC

Outputs: results/benchmark_cifar100/{results.csv, table}
"""
import os, sys, io, time, csv, copy
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import SmallCNN

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
if "--seed" in sys.argv:
    SEED = int(sys.argv[sys.argv.index("--seed") + 1])
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
BS = 64
EPOCHS = 10
LR = 1e-3
OUT = os.path.join("results", f"benchmark_cifar100_s{SEED}")
os.makedirs(OUT, exist_ok=True)

MEAN = (0.5071, 0.4867, 0.4408)   # CIFAR-100 mean/std
STD = (0.2675, 0.2565, 0.2761)

PARQUET_TRAIN = "/Users/peregrine/cifar100/cifar100/train-00000-of-00001.parquet"
PARQUET_TEST = "/Users/peregrine/cifar100/cifar100/test-00000-of-00001.parquet"

NUM_TRAIN_CLASSES = 20
FORGET = 3
HELD = [20, 21, 22]
TRAIN_CLASSES = list(range(NUM_TRAIN_CLASSES))
RETAIN_CLASSES = [c for c in TRAIN_CLASSES if c != FORGET]


def load_parquet(path):
    df = pd.read_parquet(path)
    n = len(df)
    imgs = np.empty((n, 3, 32, 32), dtype=np.uint8)
    lbls = np.empty(n, dtype=np.int64)
    for i, row in enumerate(df.itertuples()):
        b = row.img["bytes"]
        im = Image.open(io.BytesIO(b)).convert("RGB")
        imgs[i] = np.asarray(im).transpose(2, 0, 1)
        lbls[i] = int(row.fine_label)
    return imgs, lbls


print(f"Device: {DEVICE} | loading CIFAR-100 parquet ...", flush=True)
t0 = time.time()
imgs, lbls = load_parquet(PARQUET_TRAIN)
test_imgs, test_lbls = load_parquet(PARQUET_TEST)
print(f"loaded train {imgs.shape} test {test_imgs.shape} in {time.time()-t0:.0f}s")

imgs = torch.from_numpy(imgs)
lbls = torch.from_numpy(lbls)
test_imgs = torch.from_numpy(test_imgs)
test_lbls = torch.from_numpy(test_lbls)


def idx_of(l, c):
    return torch.where(l == c)[0].numpy()


train_idx = np.concatenate([idx_of(lbls, c) for c in TRAIN_CLASSES])
forget_idx = idx_of(lbls, FORGET)
retain_idx = np.concatenate([idx_of(lbls, c) for c in RETAIN_CLASSES])
held_idx = {c: idx_of(lbls, c) for c in HELD}


def loader_from(imgs, lbls, idx, shuffle=True, norm=True):
    idx = torch.from_numpy(np.asarray(idx))
    x = imgs[idx].float() / 255.0
    if norm:
        x = (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    y = lbls[idx]
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=shuffle)


train_loader = loader_from(imgs, lbls, train_idx, shuffle=True)
forget_loader = loader_from(imgs, lbls, forget_idx, shuffle=True)
retain_loader = loader_from(imgs, lbls, retain_idx, shuffle=True)
held_loaders = {c: loader_from(imgs, lbls, held_idx[c], shuffle=False) for c in HELD}
# uniform retain reference subset (5000)
r_ref = np.random.choice(retain_idx, 5000, replace=False)
retain_ref_loader = loader_from(imgs, lbls, r_ref, shuffle=False)


def train(model, loader, epochs=EPOCHS):
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    return model


def acc(model, loader):
    model.eval(); c = t = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            c += (model(x).argmax(1) == y).sum().item(); t += y.numel()
    return 100.0 * c / max(t, 1)


def probs(model, loader):
    ps = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            ps.append(torch.softmax(model(x.to(DEVICE)), 1).cpu().numpy())
    return np.concatenate(ps, 0)


def losses(model, loader):
    out = []
    crit = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            out.append(crit(model(x.to(DEVICE)), y.to(DEVICE)).cpu().numpy())
    return np.concatenate(out, 0)


def rii(pf, pr):
    M = np.stack([pf.mean(0), pr.mean(0)])
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    return float(S[1] ** 2 / (S[0] ** 2 + S[1] ** 2 + 1e-12))


def mhpr(pf, held_probs_list):
    mu_f = pf.mean(0)
    H = np.stack([p.mean(0) for p in held_probs_list])
    Hp = np.linalg.pinv(H)
    proj = Hp @ H @ mu_f
    return float(np.sum((mu_f - proj) ** 2) / (np.sum(mu_f ** 2) + 1e-12))


def mia_loss_auc(lf, lr):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(lf)), np.zeros(len(lr))])
    s = np.concatenate([-lf, -lr])
    return float(roc_auc_score(y, s))


# ---------------------------------------------------------------------------
print("training base model (10 epochs) ...", flush=True)
t0 = time.time()
base = train(SmallCNN(input_channels=3, num_classes=20).to(DEVICE), train_loader)
print(f"base trained in {time.time()-t0:.0f}s | retain_acc={acc(base,retain_loader):.1f}% "
      f"forget_acc={acc(base,forget_loader):.1f}%")


def eval_model(model):
    pf = probs(model, forget_loader)
    pr = probs(model, retain_ref_loader)
    ph = [probs(model, held_loaders[c]) for c in HELD]
    lf = losses(model, forget_loader)
    lr = losses(model, retain_ref_loader)
    return dict(
        retain_acc=acc(model, retain_loader),
        forget_acc=acc(model, forget_loader),
        rii=rii(pf, pr),
        mhpr=mhpr(pf, ph),
        mia_loss_auc=mia_loss_auc(lf, lr),
    )


results = {}
results["NoUnlearn"] = eval_model(base)

t0 = time.time()
ret = SmallCNN(input_channels=3, num_classes=20).to(DEVICE)
train(ret, retain_loader)
results["Retrain"] = eval_model(ret)
print(f"Retrain done in {time.time()-t0:.0f}s")


def grad_ascent(m, steps, lr=1e-6, loader=forget_loader):
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(m.parameters(), lr=lr)
    m.train()
    for _ in range(steps):
        x, y = next(iter(loader))
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); (-crit(m(x), y)).backward(); opt.step()
    return m


def method_neggrad(model, steps=150):
    m = copy.deepcopy(model)
    grad_ascent(m, steps)
    return m


def method_finetune(model, ascent=150, a_lr=1e-6, r_epochs=1, r_lr=1e-4):
    m = copy.deepcopy(model)
    grad_ascent(m, ascent, a_lr)
    crit = nn.CrossEntropyLoss(); opt = optim.Adam(m.parameters(), lr=r_lr)
    m.train()
    for _ in range(r_epochs):
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    return m


def method_ked(model, epochs=2, lr=1e-4):
    m = copy.deepcopy(model)
    crit = nn.CrossEntropyLoss(); opt = optim.Adam(m.parameters(), lr=lr)
    m.train(); C = 20
    uniform = torch.full((1, C), 1.0 / C, device=DEVICE)
    for _ in range(epochs):
        for x, y in forget_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            kl = nn.functional.kl_div(nn.functional.log_softmax(m(x), 1),
                                      uniform.expand(x.size(0), -1), reduction="batchmean")
            kl.backward(); opt.step()
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    return m


def method_bad_teacher(model, epochs=2, lr=1e-4):
    m = copy.deepcopy(model)
    crit = nn.CrossEntropyLoss(); opt = optim.Adam(m.parameters(), lr=lr)
    m.train()
    for _ in range(epochs):
        for x, y in forget_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            wrong = (y + 1) % 20
            opt.zero_grad(); crit(m(x), wrong).backward(); opt.step()
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    return m


for name, fn in [("NegGrad", lambda: method_neggrad(base)),
                 ("FineTune", lambda: method_finetune(base)),
                 ("KED", lambda: method_ked(base)),
                 ("BadTeacher", lambda: method_bad_teacher(base))]:
    t0 = time.time()
    m = fn()
    results[name] = eval_model(m)
    print(f"{name}: done in {time.time()-t0:.0f}s | "
          f"retain={results[name]['retain_acc']:.1f}% forget={results[name]['forget_acc']:.1f}% "
          f"rho={results[name]['rii']:.4f} MHPR={results[name]['mhpr']:.4f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"{'method':<12s} {'retain%':>7s} {'forget%':>7s} {'RII':>8s} {'MHPR':>7s} {'MIA-loss':>9s}")
print("-" * 78)
with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "retain_acc", "forget_acc", "rii", "mhpr", "mia_loss_auc"])
    for name in ["NoUnlearn", "Retrain", "NegGrad", "FineTune", "KED", "BadTeacher"]:
        r = results[name]
        print(f"{name:<12s} {r['retain_acc']:7.1f} {r['forget_acc']:7.1f} "
              f"{r['rii']:8.4f} {r['mhpr']:7.3f} {r['mia_loss_auc']:9.3f}")
        w.writerow([name, f"{r['retain_acc']:.1f}", f"{r['forget_acc']:.1f}",
                    f"{r['rii']:.6f}", f"{r['mhpr']:.6f}", f"{r['mia_loss_auc']:.4f}"])

# rank consistency with forget_acc (lower forget_acc = better forgetting)
order_forget = [n for n, _ in sorted(results.items(), key=lambda kv: kv[1]["forget_acc"])]
order_rii = [n for n, _ in sorted(results.items(), key=lambda kv: kv[1]["rii"])]
print("\nrank by forget_acc (lower=better):", order_forget)
print("rank by RII           (lower=better):", order_rii)
print("saved:", os.path.join(OUT, "results.csv"))
