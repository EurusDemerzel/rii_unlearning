#!/usr/bin/env python3
"""
Per-Sample State Disparity (PSSD) Experiment.

Computes per-sample metrics on quantized logit states for MNIST, Fashion-MNIST,
and CIFAR-10 under gradient-ascent forgetting. Compares PSSD with standard RII
and MHPR. Generates ROC curves for δ-based MIA.

Usage:
    python run_ps_experiment.py --dataset mnist --M 20 --steps 0,1,3,5,10,20,50,100
"""
import os, sys, json, argparse
import numpy as np
import torch, torch.nn as nn
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model
from metrics import (compute_rii_from_probs, quantize_logits,
                     compute_individual_state_disparity, compute_top_k_disparity,
                     compute_ps_mhpr, compute_ps_mia_roc,
                     compute_state_distribution, compute_ps_metrics_from_logits)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


def load_dataset(dataset_name):
    """Load dataset and return (all_images, all_labels, num_classes, input_shape)."""
    data_root = './data'
    if dataset_name == 'mnist':
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
        num_classes = 10
    elif dataset_name == 'fashion_mnist':
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_set = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=tfm)
        num_classes = 10
    elif dataset_name == 'cifar10':
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    all_imgs = torch.stack([train_set[i][0] for i in range(len(train_set))])
    all_lbls = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    if dataset_name in ('mnist', 'fashion_mnist'):
        all_imgs = all_imgs.squeeze(1)  # (N, 28, 28)
    return all_imgs, all_lbls, num_classes


def train_model(model, all_imgs, all_lbls, epochs=5, lr=0.001, batch_size=128):
    """Train a model on the full dataset."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(all_imgs, all_lbls),
        batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
    model.eval()
    return model


def extract_logits(model, all_imgs, batch_size=256):
    """Extract logits for all images."""
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(all_imgs), batch_size):
            batch = all_imgs[i:i+batch_size].to(DEVICE)
            all_logits.append(model(batch).cpu().numpy())
    return np.concatenate(all_logits)


def run_experiment(dataset_name, model_name, forget_class, heldout_classes,
                   n_clusters, steps_list, lr=1e-5, train_epochs=5):
    """Run full PSSD experiment on one dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}, Model: {model_name}, M={n_clusters}")
    print(f"Forget class: {forget_class}, Held-out: {heldout_classes}")

    # Load data
    all_imgs, all_lbls, num_classes = load_dataset(dataset_name)
    print(f"Data loaded: {len(all_imgs)} samples, {num_classes} classes")

    # Train base model
    model = get_model(dataset_name, DEVICE, model_name=model_name)
    model = train_model(model, all_imgs, all_lbls, epochs=train_epochs)
    print("Base model trained")

    # Get indices
    idx_f = (all_lbls == forget_class).numpy().nonzero()[0]
    idx_r = (all_lbls != forget_class).numpy().nonzero()[0]
    idx_heldout = [(all_lbls == c).numpy().nonzero()[0] for c in heldout_classes]
    print(f"Forget: {len(idx_f)}, Retain: {len(idx_r)}, Held-out: {[len(h) for h in idx_heldout]}")

    # Extract base logits and fit quantizer
    base_logits = extract_logits(model, all_imgs)
    kmeans_logits = np.concatenate([
        base_logits[idx_f], base_logits[idx_r]] + [base_logits[h] for h in idx_heldout])
    _, kmeans = quantize_logits(kmeans_logits, n_clusters=n_clusters)

    results = []
    for steps in steps_list:
        # Clone model and apply gradient ascent
        model2 = get_model(dataset_name, DEVICE, model_name=model_name)
        model2.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})

        if steps > 0:
            opt2 = torch.optim.Adam(model2.parameters(), lr=lr)
            crit = nn.CrossEntropyLoss()
            f_imgs = all_imgs[idx_f].to(DEVICE)
            f_lbls = all_lbls[idx_f].to(DEVICE)
            model2.train()
            for _ in range(steps):
                opt2.zero_grad()
                loss = -crit(model2(f_imgs), f_lbls)
                loss.backward()
                opt2.step()
        model2.eval()

        # Extract logits and quantize
        logits = extract_logits(model2, all_imgs)
        states = kmeans.predict(logits)

        states_f = states[idx_f]
        states_r = states[idx_r]
        states_h = [states[h] for h in idx_heldout]

        # State distributions
        pi_f = compute_state_distribution(states_f, n_clusters)
        pi_r = compute_state_distribution(states_r, n_clusters)

        # ─── PSSD metrics ───
        disparities, _, delta_f, delta_r, excess = \
            compute_individual_state_disparity(states_f, states_r, n_clusters)

        top_k_5 = compute_top_k_disparity(disparities, k=5)
        top_k_10 = compute_top_k_disparity(disparities, k=10)
        top_k_20 = compute_top_k_disparity(disparities, k=min(20, len(disparities)))

        # State-space RII
        M_ps = np.vstack([pi_f.reshape(1, -1), pi_r.reshape(1, -1)])
        _, S, _ = np.linalg.svd(M_ps, full_matrices=False)
        rho_S = float(S[1]**2 / (S[0]**2 + S[1]**2 + 1e-12))

        # Standard softmax RII
        # Get softmax probabilities
        probs = torch.softmax(torch.from_numpy(logits).float(), dim=1).numpy()
        probs_f = probs[idx_f]
        probs_r = probs[idx_r]
        rho_sm, mi_ub = compute_rii_from_probs(probs_f, probs_r)

        # MHPR (standard)
        H = np.array([compute_state_distribution(s, n_clusters) for s in states_h])
        K = H.shape[0]
        H_Ht_inv = np.linalg.inv(H @ H.T + 1e-12 * np.eye(K))
        alpha = H_Ht_inv @ (H @ pi_f)
        pi_f_proj = H.T @ alpha
        residual = pi_f - pi_f_proj
        rho_H_std = float(np.sum(residual**2) / (np.sum(pi_f**2) + 1e-12))

        # Per-sample MHPR
        _, rho_H_ps, _ = compute_ps_mhpr(states_f, states_h, n_clusters)

        # MIA ROC
        disparities_r = 1.0 - pi_r[states_r]
        fpr, tpr, _, mia_auc = compute_ps_mia_roc(disparities, disparities_r)

        # Softmax-based MIA (confidence = max softmax probability)
        conf_f = np.max(probs_f, axis=1)
        conf_r = np.max(probs_r, axis=1)
        try:
            from sklearn.metrics import roc_curve as roc_curve_sk, auc as auc_sk
            # Try both directions; take the better one
            fpr_sm, tpr_sm, _ = roc_curve_sk(
                np.concatenate([np.ones(len(conf_f)), np.zeros(len(conf_r))]),
                np.concatenate([conf_f, conf_r]))
            auc1 = auc_sk(fpr_sm, tpr_sm)
            fpr_sm2, tpr_sm2, _ = roc_curve_sk(
                np.concatenate([np.ones(len(conf_f)), np.zeros(len(conf_r))]),
                np.concatenate([-conf_f, -conf_r]))
            auc2 = auc_sk(fpr_sm2, tpr_sm2)
            mia_auc_sm = max(auc1, auc2, 1 - auc1, 1 - auc2)
        except Exception:
            mia_auc_sm = -1.0

        # χ² divergence
        chi2 = np.sum((pi_f - pi_r)**2 / (pi_r + 1e-12))

        result = {
            'dataset': dataset_name,
            'forget_class': int(forget_class),
            'steps': steps,
            'delta_f': delta_f,
            'delta_r': delta_r,
            'excess': excess,
            'top_k_5': top_k_5,
            'top_k_10': top_k_10,
            'top_k_20': top_k_20,
            'rho_S': rho_S,
            'rho_sm': rho_sm,
            'mi_ub': mi_ub,
            'rho_H_std': rho_H_std,
            'rho_H_ps': rho_H_ps,
            'mia_auc': mia_auc,
            'mia_auc_sm': mia_auc_sm,
            'chi2': chi2,
            'n_f': int(len(states_f)),
            'n_r': int(len(states_r)),
        }
        results.append(result)

        print(f"  steps={steps:3d} | Δ_f={delta_f:.6f} | Ψ={excess:.6f} | "
              f"ρ_S={rho_S:.6f} | ρ_sm={rho_sm:.6f} | "
              f"ρ_H_std={rho_H_std:.6f} | ρ_H_ps={rho_H_ps:.6f} | "
              f"MIA_auc={mia_auc:.4f} | MIA_sm={mia_auc_sm:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='PSSD Experiment')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist', 'cifar10'])
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--M', type=int, default=20, help='Number of states')
    parser.add_argument('--steps', type=str, default='0,1,3,5,10,20,50,100',
                        help='Comma-separated list of gradient ascent steps')
    parser.add_argument('--forget_class', type=int, default=None)
    parser.add_argument('--K', type=int, default=3, help='Number of held-out classes for MHPR')
    args = parser.parse_args()

    steps_list = [int(s) for s in args.steps.split(',')]
    np.random.seed(42)
    torch.manual_seed(42)

    # Dataset configuration
    configs = {
        'mnist': {'model': 'mlp', 'forget_class': 5, 'epochs': 5},
        'fashion_mnist': {'model': 'mlp', 'forget_class': 5, 'epochs': 5},
        'cifar10': {'model': 'cnn', 'forget_class': 3, 'epochs': 10},
    }
    cfg = configs[args.dataset]
    model_name = args.model if args.model != 'mlp' else cfg['model']
    forget_class = args.forget_class if args.forget_class is not None else cfg['forget_class']
    num_classes = 10

    # Held-out classes (K classes excluding forget class)
    available = [c for c in range(num_classes) if c != forget_class]
    heldout = available[:args.K]

    results = run_experiment(
        dataset_name=args.dataset,
        model_name=model_name,
        forget_class=forget_class,
        heldout_classes=heldout,
        n_clusters=args.M,
        steps_list=steps_list,
        train_epochs=cfg['epochs'],
    )

    # Save results
    out_dir = f"results/ps_experiment/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ps_M{args.M}_K{args.K}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Also run sensitivity analysis with different M if this is the primary run
    if args.M == 20:
        for alt_M in [50, 100]:
            alt_results = run_experiment(
                dataset_name=args.dataset,
                model_name=model_name,
                forget_class=forget_class,
                heldout_classes=heldout,
                n_clusters=alt_M,
                steps_list=steps_list,
                train_epochs=cfg['epochs'],
            )
            alt_path = os.path.join(out_dir, f"ps_M{alt_M}_K{args.K}.json")
            with open(alt_path, 'w') as f:
                json.dump(alt_results, f, indent=2)
            print(f"Saved to {alt_path}")


if __name__ == '__main__':
    main()
