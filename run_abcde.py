#!/usr/bin/env python3
"""
Supplemental experiments A→E for advisor feedback.

A: LOCO (Leave-One-Class-Out) metric on MNIST
B: Class-level forgetting on CIFAR-10 with all 4 methods
C: MMD internal representation audit
D: Multi-seed statistical validation
E: Label-flip memorization detection
"""

import os, sys, time, copy, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

# Local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model, clone_model, SimpleMLP, SmallCNN
from metrics import (
    compute_kl_divergence, compute_mutual_information,
    compute_channel_rank, compute_rii_from_probs
)
from mia import run_mia

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
    DEVICE = torch.device("mps")
    print("✅ MPS")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ CPU")

OUT_DIR = "results/supplemental"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 10

# =============================================================
# Helpers
# =============================================================
def make_loader(imgs, lbls, shuffle=True):
    ds = TensorDataset(imgs, lbls)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

def train_model(model, loader, epochs=EPOCHS, lr=LR, quiet=True):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in (range(epochs) if quiet else tqdm(range(epochs), leave=False)):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    return model

def compute_acc(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def compute_metrics(model, forget_loader, retain_loader, is_sisa=False):
    from metrics import extract_predictions
    probs_f, preds_f = extract_predictions(model, forget_loader, DEVICE, is_sisa)
    probs_r, preds_r = extract_predictions(model, retain_loader, DEVICE, is_sisa)
    nf, nr = len(probs_f), len(probs_r)
    kl_f, kl_b, kl_s = compute_kl_divergence(probs_f, probs_r)
    mi = compute_mutual_information(preds_f, preds_r, nf, nr)
    sr, sv = compute_channel_rank(probs_f, probs_r)
    rho, mi_ub = compute_rii_from_probs(probs_f, probs_r)
    mia = run_mia(model, forget_loader, retain_loader, DEVICE, is_sisa)
    return {
        'rii_rho': rho, 'mi_ub': mi_ub, 'sigma_ratio': sr,
        'kl_symmetric': kl_s, 'mutual_information': mi,
        'mia_acc': mia['mia_best_acc'], 'mia_auc': mia['mia_auc']
    }

all_results = []

def save_result(exp, method, dataset, forget_ratio, test_acc, unlearn_time, **kw):
    row = {'exp': exp, 'method': method, 'dataset': dataset,
           'forget_ratio': forget_ratio, 'test_acc': test_acc,
           'unlearn_time_sec': unlearn_time}
    row.update(kw)
    all_results.append(row)

# Helper for LOCO
def extract_predictions_loco(model, loader):
    model.eval()
    all_p, all_pr = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            pr = torch.softmax(model(x), 1).cpu().numpy()
            pd = pr.argmax(1)
            all_p.append(pr); all_pr.append(pd)
    return np.concatenate(all_p), np.concatenate(all_pr)

# =============================================================
# A: LOCO on MNIST
# =============================================================
print("\n" + "="*60)
print("A: LOCO (Leave-One-Class-Out) on MNIST")
print("="*60)

# Load MNIST
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_train = datasets.MNIST(root='./data', train=True, download=True,
                               transform=transform_mnist)
mnist_test  = datasets.MNIST(root='./data', train=False, download=True,
                               transform=transform_mnist)
all_imgs = mnist_train.data.float() / 255.0
all_imgs = (all_imgs - 0.1307) / 0.3081
all_lbls = mnist_train.targets               # (60000,)

# Separate indices: class 0 = held-out, class 5 = forget
idx_heldout = (all_lbls == 0).nonzero().squeeze().tolist()   # ~6000
idx_class5  = (all_lbls == 5).nonzero().squeeze().tolist()   # ~6000
idx_retain  = (all_lbls > 0).nonzero().squeeze().tolist()      # 54000
# For training on classes 1-9 only: remove class 5 from training
idx_train   = [i for i in idx_retain if i not in idx_class5]   # 48000

# Make loaders
def _loader(idx, shuffle=True):
    imgs = all_imgs[idx].unsqueeze(1)
    lbls = all_lbls[idx]
    return make_loader(imgs, lbls, shuffle)

heldout_loader = _loader(idx_heldout, shuffle=False)
forget_loader  = _loader(idx_class5, shuffle=False)
retain_loader  = _loader([i for i in idx_retain if i not in idx_class5], shuffle=False)
train_loader   = _loader(idx_train, shuffle=True)

# ---- A1: NoUnlearn - baseline (trained on classes 1-9 only, no class 5 seen) ----
print("A1: NoUnlearn (baseline)...")
model = get_model('mnist', DEVICE)
model = train_model(model, train_loader, epochs=EPOCHS)
acc = compute_acc(model, make_loader(mnist_test.data.float()/255.0,
                                      mnist_test.targets, shuffle=False))

# LOCO comparison: forget(class5) vs heldout(class0)
probs_f, _ = extract_predictions_loco(model, forget_loader)
probs_h, _ = extract_predictions_loco(model, heldout_loader)
rho_loco, mi_loco = compute_rii_from_probs(probs_f, probs_h)

# Standard RII: forget(class5) vs retain(classes 1-9 minus 5)
probs_r, _ = extract_predictions_loco(model, retain_loader)
rho_std, mi_std = compute_rii_from_probs(probs_f, probs_r)

print(f"  Acc={acc*100:.2f}% | LOCO-RII={rho_loco:.6f} | Std-RII={rho_std:.6f}")
save_result('A_LOCO', 'NoUnlearn', 'MNIST', 1.0, acc, 0,
            loco_rii=rho_loco, standard_rii=rho_std)

# ---- A2: Retrain from scratch ----
print("A2: Retrain from scratch on retain set...")
t0 = time.time()
model2 = get_model('mnist', DEVICE)
model2 = train_model(model2, _loader([i for i in idx_retain if i not in idx_class5], shuffle=True),
                      epochs=EPOCHS)
t_rt = time.time() - t0
acc2 = compute_acc(model2, make_loader(mnist_test.data.float()/255.0,
                                        mnist_test.targets, shuffle=False))

probs_f2, _ = extract_predictions_loco(model2, forget_loader)
probs_h2, _ = extract_predictions_loco(model2, heldout_loader)
rho_loco2, _ = compute_rii_from_probs(probs_f2, probs_h2)
probs_r2, _ = extract_predictions_loco(model2, retain_loader)
rho_std2, _ = compute_rii_from_probs(probs_f2, probs_r2)

print(f"  Acc={acc2*100:.2f}% | LOCO-RII={rho_loco2:.6f} | Std-RII={rho_std2:.6f} | Time={t_rt:.1f}s")
save_result('A_LOCO', 'Retrain', 'MNIST', 1.0, acc2, t_rt,
            loco_rii=rho_loco2, standard_rii=rho_std2)

# ---- A3: FineTune - gradient ascent on class 5 ----
print("A3: FineTune (ascent on class 5)...")
model3 = clone_model(model, 'mnist', DEVICE)  # start from baseline model
t0 = time.time()
ft_opt = optim.Adam(model3.parameters(), lr=1e-5)
ft_crit = nn.CrossEntropyLoss()
model3.train()
for _ in range(5):
    for x, y in forget_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        ft_opt.zero_grad()
        (-ft_crit(model3(x), y)).backward()
        ft_opt.step()
t_ft = time.time() - t0
acc3 = compute_acc(model3, make_loader(mnist_test.data.float()/255.0,
                                        mnist_test.targets, shuffle=False))

probs_f3, _ = extract_predictions_loco(model3, forget_loader)
probs_h3, _ = extract_predictions_loco(model3, heldout_loader)
rho_loco3, _ = compute_rii_from_probs(probs_f3, probs_h3)
probs_r3, _ = extract_predictions_loco(model3, retain_loader)
rho_std3, _ = compute_rii_from_probs(probs_f3, probs_r3)

print(f"  Acc={acc3*100:.2f}% | LOCO-RII={rho_loco3:.6f} | Std-RII={rho_std3:.6f} | Time={t_ft:.1f}s")
save_result('A_LOCO', 'FineTune', 'MNIST', 1.0, acc3, t_ft,
            loco_rii=rho_loco3, standard_rii=rho_std3)

# =============================================================
# B: Class-level forgetting on CIFAR-10 (cat class)
# =============================================================
print("\n" + "="*60)
print("B: Class-level forgetting (cat class) on CIFAR-10")
print("="*60)

transform_c10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
c10_train = datasets.CIFAR10(root='./data', train=True, download=False,
                               transform=transform_c10)
c10_test  = datasets.CIFAR10(root='./data', train=False, download=False,
                               transform=transform_c10)

# Extract tensors
c10_imgs = torch.stack([c10_train[i][0] for i in range(len(c10_train))])
c10_lbls = torch.tensor(c10_train.targets)

cat_class = 3  # "cat" in CIFAR-10
idx_cat   = (c10_lbls == cat_class).nonzero().squeeze().tolist()
idx_other = [i for i in range(len(c10_train)) if i not in idx_cat]

# Keep data on CPU for DataLoader creation
cat_loader    = make_loader(c10_imgs[idx_cat], c10_lbls[idx_cat], shuffle=False)
other_loader  = make_loader(c10_imgs[idx_other], c10_lbls[idx_other], shuffle=False)
other_train   = make_loader(c10_imgs[idx_other], c10_lbls[idx_other], shuffle=True)

# Test loader
c10_test_imgs = torch.stack([c10_test[i][0] for i in range(len(c10_test))])
c10_test_lbls = torch.tensor(c10_test.targets)
test_loader    = make_loader(c10_test_imgs, c10_test_lbls, shuffle=False)

def _c10_model():
    return get_model('cifar10', DEVICE, model_name='cnn')

def run_class_unlearn(method, model_orig=None):
    """Run one class-level unlearning experiment and return metrics."""
    t0 = time.time()

    if method == 'NoUnlearn':
        full_loader = make_loader(c10_imgs, c10_lbls, shuffle=True)
        m = _c10_model()
        m = train_model(m, full_loader, epochs=EPOCHS)
        acc = compute_acc(m, test_loader)
        elapsed = 0
        # Compute RII metrics for NoUnlearn
        from metrics import extract_predictions
        probs_f, preds_f = extract_predictions(m, cat_loader, DEVICE)
        probs_r, preds_r = extract_predictions(m, other_loader, DEVICE)
        rho, mi_ub = compute_rii_from_probs(probs_f, probs_r)
        sr, sv = compute_channel_rank(probs_f, probs_r)
        return {'meth': method, 'acc': acc, 'rii_rho': rho, 'mi_ub': mi_ub,
                'sigma_ratio': sr, 'unlearn_time': elapsed, 'model': m}

    elif method == 'Retrain':
        # Retrain on non-cat classes only
        m = _c10_model()
        m = train_model(m, other_train, epochs=EPOCHS)
        acc = compute_acc(m, test_loader)
        elapsed = time.time() - t0

    elif method == 'SISA':
        from unlearn import unlearn_sisa
        # Use SISA on all data with cat class as forget set
        models_sisa, elapsed = unlearn_sisa(
            get_model, 'cifar10', DEVICE,
            c10_imgs, c10_lbls,
            forget_indices=idx_cat, retain_indices=idx_other,
            N_total=len(c10_train),
            sisa_num_shards=5, sisa_slices_per_shard=5,
            sisa_epochs_per_slice=1, lr=LR, batch_size=BATCH_SIZE,
            model_name='cnn'
        )
        from metrics import extract_predictions
        probs_f, preds_f = extract_predictions(models_sisa, cat_loader, DEVICE, is_sisa=True)
        probs_r, preds_r = extract_predictions(models_sisa, other_loader, DEVICE, is_sisa=True)
        rho, mi_ub = compute_rii_from_probs(probs_f, probs_r)
        sr, sv = compute_channel_rank(probs_f, probs_r)
        return {
            'meth': method, 'acc': None,
            'rii_rho': rho, 'mi_ub': mi_ub, 'sigma_ratio': sr,
            'unlearn_time': elapsed
        }

    else:  # FineTune
        from unlearn import unlearn_finetune
        m_un, elapsed = unlearn_finetune(
            model_orig, 'cifar10', DEVICE,
            c10_imgs, c10_lbls,
            forget_indices=idx_cat, retain_indices=idx_other,
            finetune_epochs=3, finetune_lr=1e-5,
            finetune_train_on_retain=False, batch_size=BATCH_SIZE,
            model_name='cnn'
        )
        m = m_un
        acc = compute_acc(m, test_loader)

    if method != 'SISA':
        from metrics import extract_predictions
        probs_f, preds_f = extract_predictions(m, cat_loader, DEVICE)
        probs_r, preds_r = extract_predictions(m, other_loader, DEVICE)
        rho, mi_ub = compute_rii_from_probs(probs_f, probs_r)
        sr, sv = compute_channel_rank(probs_f, probs_r)

    return {
        'meth': method, 'acc': acc if method != 'SISA' else None,
        'rii_rho': rho, 'mi_ub': mi_ub, 'sigma_ratio': sr,
        'unlearn_time': elapsed
    }

methods_b = ['NoUnlearn', 'Retrain', 'SISA', 'FineTune']
orig_model = None
for meth in methods_b:
    print(f"B: {meth}...")
    if meth == 'FineTune':
        res = run_class_unlearn(meth, model_orig=orig_model)
    else:
        res = run_class_unlearn(meth)
    if meth == 'NoUnlearn' and res.get('model'):
        orig_model = res['model']
    acc_str = f"acc={res['acc']*100:.2f}%" if res.get('acc') is not None else "acc=N/A"
    rho_str = f"ρ={res.get('rii_rho', 0):.6f}" if res.get('rii_rho') is not None else "ρ=N/A"
    sr_str = f"σ₂/σ₁={res.get('sigma_ratio', 0):.6f}" if res.get('sigma_ratio') is not None else "σ₂/σ₁=N/A"
    print(f"  {rho_str} | {sr_str} | {acc_str} | time={res['unlearn_time']:.1f}s")
    save_result('B_class', res['meth'], 'CIFAR10', 1.0, res.get('acc'),
                res['unlearn_time'],
                rii_rho=res.get('rii_rho', 0),
                sigma_ratio=res.get('sigma_ratio', 0))

# =============================================================
# C: MMD Internal Representation Audit
# =============================================================
print("\n" + "="*60)
print("C: MMD Internal Representation Audit")
print("="*60)

def compute_mmd(model, loader_f, loader_r, device, layer=-1, gamma='auto'):
    """
    Compute MMD between penultimate-layer features of forget and retain sets.
    Uses a simple RBF kernel.
    """
    model.eval()
    def extract_features(loader):
        features = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                # Forward to penultimate layer
                # For SimpleMLP: stop before fc2
                # For SmallCNN: stop before fc1
                if hasattr(model, 'fc1') and not hasattr(model, 'conv1'):
                    # SimpleMLP
                    h = x.view(x.size(0), -1)
                    h = model.relu(model.fc1(h))
                elif hasattr(model, 'conv1'):
                    # SmallCNN
                    h = model.relu(model.bn1(model.conv1(x)))
                    h = model.pool(h)
                    h = model.relu(model.bn2(model.conv2(h)))
                    h = model.pool(h)
                    h = model.relu(model.bn3(model.conv3(h)))
                    h = model.pool(h)
                    h = h.view(h.size(0), -1)
                    h = model.relu(model.fc1(h))
                features.append(h.cpu().numpy())
        return np.concatenate(features, axis=0)

    H_f = extract_features(loader_f)
    H_r = extract_features(loader_r)

    # RBF kernel MMD
    n_f, n_r = len(H_f), len(H_r)
    if gamma == 'auto':
        gamma = 1.0 / H_f.shape[1]

    # Compute kernel matrices
    def rbf(x, y, g):
        dist = np.sum(x**2, 1, keepdims=True) + np.sum(y**2, 1, keepdims=True).T - 2 * x @ y.T
        return np.exp(-g * dist)

    K_ff = rbf(H_f, H_f, gamma)
    K_rr = rbf(H_r, H_r, gamma)
    K_fr = rbf(H_f, H_r, gamma)

    mmd = (K_ff.sum() - np.trace(K_ff)) / (n_f * (n_f - 1)) \
        + (K_rr.sum() - np.trace(K_rr)) / (n_r * (n_r - 1)) \
        - 2 * K_fr.sum() / (n_f * n_r)
    return float(mmd)

# Use MNIST model from experiment A (NoUnlearn baseline model)
print("C: Computing MMD on LOCO experiment models...")
for name, m in [('NoUnlearn', model), ('Retrain', model2), ('FineTune', model3)]:
    mmd = compute_mmd(m, forget_loader, heldout_loader, DEVICE)
    print(f"  {name}: MMD={mmd:.6f}")

# =============================================================
# D: Multi-seed statistical validation
# =============================================================
print("\n" + "="*60)
print("D: Multi-seed validation (MNIST 10%, 5 seeds)")
print("="*60)

def run_single_seed(dataset, method, n_f, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if dataset == 'mnist':
        imgs, lbls = all_imgs, all_lbls
        model_fn = lambda: get_model('mnist', DEVICE)
        test_ds = datasets.MNIST(root='./data', train=False, download=False,
                                  transform=transform_mnist)
        test_imgs = test_ds.data.float()/255.0
        test_imgs = (test_imgs - 0.1307) / 0.3081
        test_lbls = test_ds.targets
        C = 10
    else:  # cifar10
        imgs, lbls = c10_imgs, c10_lbls
        model_fn = lambda: get_model('cifar10', DEVICE)
        test_lbls = c10_test_lbls
        test_imgs = c10_test_imgs
        C = 10

    N = len(imgs)
    perm = torch.randperm(N)
    forget_idx = perm[:n_f].tolist()
    retain_idx = perm[n_f:].tolist()

    forget_ld = make_loader(imgs[forget_idx], lbls[forget_idx], shuffle=False)
    retain_ld = make_loader(imgs[retain_idx], lbls[retain_idx], shuffle=False)
    full_ld   = make_loader(imgs, lbls, shuffle=True)

    t0 = time.time()
    if method == 'NoUnlearn':
        m = model_fn()
        m = train_model(m, full_ld)
        elapsed = 0
    elif method == 'Retrain':
        m = model_fn()
        m = train_model(m, make_loader(imgs[retain_idx], lbls[retain_idx], shuffle=True))
        elapsed = time.time() - t0
    elif method == 'SISA':
        from unlearn import unlearn_sisa
        models_sisa, elapsed = unlearn_sisa(
            get_model, dataset, DEVICE, imgs, lbls,
            forget_indices=forget_idx, retain_indices=retain_idx, N_total=N,
            sisa_num_shards=5, sisa_slices_per_shard=10,
            sisa_epochs_per_slice=1, lr=LR, batch_size=BATCH_SIZE
        )
        probs_f, _ = extract_preds_sisa(models_sisa, forget_ld, DEVICE)
        probs_r, _ = extract_preds_sisa(models_sisa, retain_ld, DEVICE)
        rho, mi_ub = compute_rii_from_probs(probs_f, probs_r)
        test_ld = make_loader(test_imgs.to(DEVICE), test_lbls.to(DEVICE), shuffle=False)
        return {'method': method, 'seed': seed, 'rii_rho': rho,
                'test_acc': None, 'unlearn_time': elapsed,
                'exp': 'D_multiseed', 'dataset': dataset,
                'forget_ratio': n_f / N}

    forget_ratio = n_f / N
    test_ld = make_loader(test_imgs.to(DEVICE), test_lbls.to(DEVICE), shuffle=False)
    acc = compute_acc(m, test_ld)
    probs_f, _ = extract_preds(m, forget_ld, DEVICE)
    probs_r, _ = extract_preds(m, retain_ld, DEVICE)
    rho, mi_ub = compute_rii_from_probs(probs_f, probs_r)

    return {'method': method, 'seed': seed, 'rii_rho': rho,
            'test_acc': acc, 'unlearn_time': elapsed,
            'exp': 'D_multiseed', 'dataset': dataset,
            'forget_ratio': forget_ratio}

def extract_preds(model, loader, device):
    model.eval()
    all_p = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            all_p.append(torch.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(all_p), None

def extract_preds_sisa(models, loader, device):
    all_p = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            probs = torch.zeros(x.size(0), 10, device=device)
            for m in models: probs += torch.softmax(m(x), 1)
            probs /= len(models)
            all_p.append(probs.cpu().numpy())
    return np.concatenate(all_p), None

for seed in [42, 123, 456, 789, 101112]:
    for meth in ['NoUnlearn', 'Retrain']:
        res = run_single_seed('mnist', meth, 6000, seed)
        print(f"  seed={seed} {meth}: ρ={res['rii_rho']:.6f}")
        save_result(res['exp'], res['method'], res['dataset'],
                res.get('forget_ratio', 0.1), res.get('test_acc'),
                res.get('unlearn_time', 0), rii_rho=res['rii_rho'])

# =============================================================
# E: Label-flip memorization detection
# =============================================================
print("\n" + "="*60)
print("E: Label-flip memorization detection")
print("="*60)

# Take MNIST, flip 5% of random labels
labels_flipped = all_lbls.clone()
flip_rate = 0.05
n_flip = int(len(labels_flipped) * flip_rate)
flip_idx = torch.randperm(len(labels_flipped))[:n_flip]
labels_flipped[flip_idx] = torch.randint(0, 10, (n_flip,))

# Train model on flipped labels
full_flipped = make_loader(all_imgs.unsqueeze(1), labels_flipped, shuffle=True)
model_flip = get_model('mnist', DEVICE)
model_flip = train_model(model_flip, full_flipped)

# RII between flipped vs non-flipped samples
flip_set = set(flip_idx.tolist())
nonflip_set = set(range(len(all_lbls))) - flip_set

flip_loader   = make_loader(all_imgs[list(flip_set)].unsqueeze(1),
                              labels_flipped[list(flip_set)], shuffle=False)
nonflip_loader = make_loader(all_imgs[list(nonflip_set)].unsqueeze(1),
                               labels_flipped[list(nonflip_set)], shuffle=False)

probs_f, _ = extract_preds(model_flip, flip_loader, DEVICE)
probs_r, _ = extract_preds(model_flip, nonflip_loader, DEVICE)
rho_flip, _ = compute_rii_from_probs(probs_f, probs_r)
sr, sv = compute_channel_rank(probs_f, probs_r)

print(f"  Label-flip RII={rho_flip:.6f} | σ₂/σ₁={sr:.6f}")
save_result('E_labelflip', 'NoUnlearn', 'MNIST', flip_rate,
             compute_acc(model_flip, make_loader(all_imgs.unsqueeze(1),
                          all_lbls, shuffle=False)), 0,
            rii_rho=rho_flip, sigma_ratio=sr)

# =============================================================
# Save all results
# =============================================================
import pandas as pd
df = pd.DataFrame(all_results)
csv_path = os.path.join(OUT_DIR, "results_abcde.csv")
df.to_csv(csv_path, index=False)
print(f"\n✅ All results saved to {csv_path}")
print(df.to_string(index=False))
