#!/usr/bin/env python3
"""Gradient sweep: RII vs forgetting intensity on CIFAR-10"""
import torch, torch.nn as nn, torch.optim as optim, numpy as np, csv, sys
sys.path.insert(0, '/Users/peregrine/one_rank')
from models import get_model
from pipeline import load_dataset

device = torch.device("mps")
ds_name, model_name = "cifar10", "cnn"
all_imgs, all_lbls, N, test_ldr, _ = load_dataset(ds_name)

print("Training base model...")
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

Nf = int(N * 0.05)
np.random.seed(42)
f_idx = torch.from_numpy(np.random.choice(N, Nf, replace=False))
r_idx = torch.tensor([i for i in range(N) if i not in f_idx.tolist()])

results = []
for grad_epochs in [0, 1, 3, 10, 50]:
    m = get_model(ds_name, device, model_name)
    m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    if grad_epochs > 0:
        opt2 = optim.Adam(m.parameters(), lr=0.00001)
        m.train()
        for _ in range(grad_epochs):
            perm = torch.randperm(Nf)
            for i in range(0, Nf, bs):
                idx = f_idx[perm[i:i+bs]]
                x, y = all_imgs[idx].to(device), all_lbls[idx].to(device)
                opt2.zero_grad(); (-crit(m(x), y)).backward(); opt2.step()

    m.eval()
    with torch.no_grad():
        def avg_prob(indices):
            probs = []
            for i in range(0, len(indices), bs):
                x = all_imgs[indices[i:i+bs]].to(device)
                probs.append(torch.softmax(m(x).float(), dim=-1).cpu().numpy())
            return np.concatenate(probs).mean(0)
        mu_f = avg_prob(f_idx)
        mu_r = avg_prob(r_idx[:5000])
        M = np.stack([mu_f, mu_r])
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        rho = S[1]**2/(S[0]**2+S[1]**2)

    correct = sum((m(x.to(device)).argmax(1) == y.to(device)).sum().item()
                  for x, y in test_ldr)
    acc = 100*correct/len(test_ldr.dataset)
    results.append((grad_epochs, rho, S[1]/S[0], acc))
    print(f"  steps={grad_epochs:3d}  rho={rho:.6e}  acc={acc:.1f}%")

with open("/Users/peregrine/one_rank/gradient_sweep_results.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["grad_epochs", "rho", "sigma_ratio", "test_acc"])
    w.writerows(results)
print("\nDone!")
