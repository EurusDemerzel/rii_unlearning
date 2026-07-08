#!/usr/bin/env python3
"""Targeted forgetting: 遗忘CIFAR-10中整个cat类，观察RII随遗忘强度的变化"""
import torch, torch.nn as nn, torch.optim as optim, numpy as np, csv, sys
sys.path.insert(0, '/Users/peregrine/one_rank')
from models import get_model
from pipeline import load_dataset

device = torch.device("mps")
ds_name, model_name = "cifar10", "cnn"
all_imgs, all_lbls, N, test_ldr, _ = load_dataset(ds_name)

# Train base model
print("Training base model (CIFAR-10 CNN)...")
model = get_model(ds_name, device, model_name)
opt = optim.Adam(model.parameters(), lr=0.001)
crit = nn.CrossEntropyLoss()
bs = 64
for ep in range(10):
    perm = torch.randperm(N)
    for i in range(0, N, bs):
        idx = perm[i:i+bs]
        x, y = all_imgs[idx].to(device), all_lbls[idx].to(device)
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    print(f"  Epoch {ep+1}/10")

model.eval()
with torch.no_grad():
    correct = sum((model(x.to(device)).argmax(1) == y.to(device)).sum().item()
                  for x, y in test_ldr)
    base_acc = 100*correct/len(test_ldr.dataset)
print(f"Base test acc: {base_acc:.1f}%")

# Identify cat class (class 3 in CIFAR-10)
CAT_CLASS = 3
cat_idx = torch.where(all_lbls == CAT_CLASS)[0]
noncat_idx = torch.where(all_lbls != CAT_CLASS)[0]
Nf = len(cat_idx)
print(f"Forget set: {Nf} cat samples (class {CAT_CLASS})")
print(f"Retain set: {len(noncat_idx)} non-cat samples")

# Subsample retain to 5000 for speed
np.random.seed(42)
r_sample = noncat_idx[torch.from_numpy(np.random.choice(len(noncat_idx), 5000, replace=False))]

def compute_rii_avgprob(m, f_idx, r_idx):
    """Compute RII using avg probability vectors"""
    m.eval()
    with torch.no_grad():
        def avg_p(indices):
            ps = []
            for i in range(0, len(indices), bs):
                x = all_imgs[indices[i:i+bs]].to(device)
                ps.append(torch.softmax(m(x).float(), dim=-1).cpu().numpy())
            return np.concatenate(ps).mean(0)
        M = np.stack([avg_p(f_idx), avg_p(r_idx)])
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        rho = S[1]**2/(S[0]**2+S[1]**2)
        return rho, S[1]/S[0]

def compute_mia(m, f_idx, r_idx):
    """Simple loss-threshold MIA"""
    m.eval()
    with torch.no_grad():
        def losses(indices):
            ls = []
            for i in range(0, len(indices), bs):
                x = all_imgs[indices[i:i+bs]].to(device)
                y = all_lbls[indices[i:i+bs]].to(device)
                logits = m(x)
                ls.extend(nn.CrossEntropyLoss(reduction='none')(logits, y).cpu().tolist())
            return np.array(ls)
        l_f = losses(f_idx[:2000])
        l_r = losses(r_idx[:2000])
        # Simple threshold: median of retain losses
        tau = np.median(l_r)
        mia_acc = (np.mean(l_f < tau) + np.mean(l_r >= tau)) / 2
        return mia_acc

def test_acc(m):
    m.eval()
    with torch.no_grad():
        correct = sum((m(x.to(device)).argmax(1) == y.to(device)).sum().item()
                      for x, y in test_ldr)
    return 100*correct/len(test_ldr.dataset)

# Baseline
rho0, sr0 = compute_rii_avgprob(model, cat_idx, r_sample)
mia0 = compute_mia(model, cat_idx, r_sample)
print(f"\n{'='*50}")
print(f"Baseline:  rho={rho0:.2e}  mia={mia0:.3f}  acc={base_acc:.1f}%")

# Gradient ascent sweep
results = [("baseline", 0, rho0, sr0, mia0, base_acc)]

for steps in [1, 3, 10, 30, 100]:
    # Clone and attack
    m = get_model(ds_name, device, model_name)
    m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    m.train()
    opt2 = optim.Adam(m.parameters(), lr=1e-6)

    # Gradient ascent on cat class
    for _ in range(steps):
        perm = torch.randperm(Nf)
        for i in range(0, Nf, bs):
            idx = cat_idx[perm[i:i+bs]]
            x, y = all_imgs[idx].to(device), all_lbls[idx].to(device)
            opt2.zero_grad(); (-crit(m(x), y)).backward(); opt2.step()

    acc = test_acc(m)
    rho, sr = compute_rii_avgprob(m, cat_idx, r_sample)
    mia = compute_mia(m, cat_idx, r_sample)

    results.append((f"ascent_{steps}", steps, rho, sr, mia, acc))
    print(f"  ascent_{steps:3d}:  rho={rho:.2e}  mia={mia:.3f}  acc={acc:.1f}%")

# Also try: ascent + finetune on retain (to recover accuracy)
print(f"\n{'='*50}")
print("Ascent + FineTune (recover accuracy):")
for steps in [10, 20, 50]:
    m = get_model(ds_name, device, model_name)
    m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    m.train()
    opt2 = optim.Adam(m.parameters(), lr=1e-6)

    # Gradient ascent on cat
    for _ in range(steps):
        perm = torch.randperm(Nf)
        for i in range(0, Nf, bs):
            idx = cat_idx[perm[i:i+bs]]
            x, y = all_imgs[idx].to(device), all_lbls[idx].to(device)
            opt2.zero_grad(); (-crit(m(x), y)).backward(); opt2.step()

    # FineTune on retain (3 epochs, lr=1e-4)
    for _ in range(3):
        perm = torch.randperm(len(r_sample))
        for i in range(0, len(r_sample), bs):
            idx = r_sample[perm[i:i+bs]]
            x, y = all_imgs[idx].to(device), all_lbls[idx].to(device)
            opt2.zero_grad(); crit(m(x), y).backward(); opt2.step()

    acc = test_acc(m)
    rho, sr = compute_rii_avgprob(m, cat_idx, r_sample)
    mia = compute_mia(m, cat_idx, r_sample)

    results.append((f"ascent{steps}+ft", steps, rho, sr, mia, acc))
    print(f"  ascent{steps}+ft:  rho={rho:.2e}  mia={mia:.3f}  acc={acc:.1f}%")

# Save
with open("/Users/peregrine/one_rank/targeted_forgetting_results.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["experiment", "steps", "rho", "sigma_ratio", "mia", "test_acc"])
    w.writerows(results)

print(f"\n{'='*50}")
print("TARGETED FORGETTING RESULTS (forgetting entire cat class)")
print(f"{'='*50}")
for exp, st, rho, sr, mia, acc in results:
    print(f"  {exp:16s}  rho={rho:8.2e}  mia={mia:.3f}  acc={acc:.1f}%")
print("\nDone!")
