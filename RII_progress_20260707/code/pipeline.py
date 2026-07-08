#!/usr/bin/env python3
"""
Complete Machine Unlearning Experiment Pipeline
=================================================

Unified CLI for running machine unlearning experiments across:
  - Datasets:    MNIST, CIFAR-10
  - Methods:     NoUnlearning, Retrain, SISA, FineTune
  - Forget ratios: 0.01, 0.05, 0.10, 0.20

Outputs:
  - results.csv          full metrics table
  - *.png                 four visualisation plots
  - models/               saved model checkpoints (optional)
"""

import argparse
import os
import sys
import time
import pickle
import random
import json
import copy
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# Local modules
from models  import get_model, clone_model, SimpleMLP, SmallCNN
from unlearn import unlearn_none, unlearn_retrain, unlearn_sisa, unlearn_finetune, unlearn_rii_driven
from metrics import compute_all_metrics
from mia    import run_mia
from visualize import generate_all_plots


# =============================================================================
# 0.  Seed & device
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("✅  Apple Metal (MPS) acceleration enabled.")
        return torch.device("mps")
    elif torch.backends.mps.is_built():
        print("⚠️  MPS built but unavailable → CPU fallback.")
        return torch.device("cpu")
    else:
        print("⚠️  MPS not available → CPU.")
        return torch.device("cpu")


# =============================================================================
# 1.  CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Machine Unlearning Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --dataset mnist --unlearn_method all --forget_ratio 0.1
  python pipeline.py --dataset cifar10 --unlearn_method retrain sisa --forget_ratio 0.05 0.1 0.2
  python pipeline.py --dataset mnist --unlearn_method all --forget_ratio 0.1 --no_save
        """,
    )
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10", "cifar100"],
                   help="Dataset (default: mnist)")
    p.add_argument("--unlearn_method", nargs="+", default=["all"],
                   choices=["none", "retrain", "sisa", "finetune", "rii", "all",
                            "NoUnlearning", "Retrain", "SISA", "FineTune", "RII"],
                   help="Unlearning method(s) (default: all)")
    p.add_argument("--forget_ratio", nargs="+", type=float,
                   default=[0.01, 0.05, 0.10, 0.20],
                   help="Forget ratio(s) (default: 0.01 0.05 0.1 0.2)")
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs (default: 10)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: results/run_<timestamp>)")
    p.add_argument("--no_save", action="store_true",
                   help="Do NOT save model checkpoints")
    p.add_argument("--config", default="config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--model", default="mlp",
                   choices=["mlp", "mlp2", "mlp3", "deepmlp", "cnn"],
                   help="Model architecture (default: mlp)")
    p.add_argument("--num_seeds", type=int, default=1,
                   help="Number of random seeds for error bars (default: 1)")
    p.add_argument("--subset", type=int, default=0,
                   help="Use only N training samples (0 = full dataset)")
    p.add_argument("--finetune_variant", default="ascent+descent",
                   choices=["ascent+descent", "ascent_only"],
                   help="FineTune variant: ascent+descent (safe) or ascent_only (pure forgetting)")
    p.add_argument("--lambda_rii", type=float, default=1.0,
                   help="RII loss weight for RII-driven unlearning (default: 1.0)")
    p.add_argument("--rii_tau", type=float, default=0.01,
                   help="Early-stop threshold for RII-driven unlearning (default: 0.01)")
    p.add_argument("--rii_epochs", type=int, default=10,
                   help="Max epochs for RII-driven unlearning (default: 10)")
    return p.parse_args()


# =============================================================================
# 2.  Data loading  (with robust CIFAR-10 download)
# =============================================================================
def _download_cifar10(data_root: str, max_retries: int = 3):
    """
    Download CIFAR-10 with retry + multiple mirrors.
    Returns (train_ds, test_ds) once successful.
    """
    import urllib.request
    import tarfile

    cifar_dir = os.path.join(data_root, "cifar-10-batches-py")
    if os.path.isdir(cifar_dir) and len(os.listdir(cifar_dir)) >= 6:
        # Already downloaded — skip
        return

    mirrors = [
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz",
    ]
    tgz_path = os.path.join(data_root, "cifar-10-python.tar.gz")

    for attempt in range(max_retries):
        for mirror in mirrors:
            try:
                print(f"   Downloading CIFAR-10 from {mirror} (attempt {attempt+1}/{max_retries})...")
                urllib.request.urlretrieve(mirror, tgz_path)
                # Extract
                with tarfile.open(tgz_path, "r:gz") as tar:
                    tar.extractall(path=data_root)
                os.remove(tgz_path)
                print("   ✅  CIFAR-10 downloaded & extracted.")
                return
            except Exception as e:
                print(f"   ⚠️  Mirror {mirror} failed: {e}")
                if os.path.exists(tgz_path):
                    os.remove(tgz_path)
        if attempt < max_retries - 1:
            wait = 2 ** attempt * 10
            print(f"   Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(
        "CIFAR-10 download failed after all retries.\n"
        "Please download manually:\n"
        "  curl -L -o data/cifar-10-python.tar.gz "
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n"
        "  tar -xzf data/cifar-10-python.tar.gz -C data/\n"
        "Then re-run the pipeline."
    )


def load_dataset(name: str, data_root: str = "./data"):
    """Load MNIST or CIFAR-10.  Returns (all_images, all_labels, N, test_loader, test_ds)."""
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(root=data_root, train=True,  download=True,
                                   transform=transform)
        test_ds  = datasets.MNIST(root=data_root, train=False, download=True,
                                   transform=transform)
    elif name == "cifar10":
        _download_cifar10(data_root)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10(root=data_root, train=True,  download=False,
                                     transform=transform_train)
        test_ds  = datasets.CIFAR10(root=data_root, train=False, download=False,
                                     transform=transform_test)
    elif name == "cifar100":
        import numpy as np
        raw_data, raw_labels = [], []
        for b in range(1, 6):
            with open(os.path.join(data_root, "cifar-100-python", f"data_batch_{b}"), "rb") as f:
                batch = pickle.load(f)
            raw_data.append(np.array(batch["data"]))
            raw_labels.append(np.array(batch["fine_labels"]))
        all_data = np.concatenate(raw_data).reshape(-1, 3, 32, 32) / 255.0
        all_images = torch.tensor(all_data, dtype=torch.float32)
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1,3,1,1)
        std  = torch.tensor([0.2675, 0.2565, 0.2761]).view(1,3,1,1)
        all_images = (all_images - mean) / std
        all_labels = torch.tensor(np.concatenate(raw_labels), dtype=torch.long)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        # Load test from pickle too (torchvision format incompatible)
        with open(os.path.join(data_root, "cifar-100-python", "test_batch"), "rb") as f:
            test_batch = pickle.load(f)
        test_data = np.array(test_batch["data"]).reshape(-1, 3, 32, 32) / 255.0
        test_imgs = torch.tensor(test_data, dtype=torch.float32)
        test_imgs = (test_imgs - mean) / std
        test_labels = torch.tensor(np.array(test_batch["fine_labels"]), dtype=torch.long)
        test_ds = TensorDataset(test_imgs, test_labels)
        N = len(all_labels)
        print(f"   Train: {N:,}  |  Test: {len(test_ds):,}  |  Classes: 100")
        return all_images, all_labels, N, DataLoader(test_ds, batch_size=64, shuffle=False), test_ds
    else:
        raise ValueError(f"Unknown dataset: {name}")

    bs = 64
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    # Extract raw tensors for manual index-based splitting
    if name == "mnist":
        all_images = train_ds.data.float() / 255.0
        all_images = (all_images - 0.1307) / 0.3081
        all_labels = train_ds.targets
    elif name == "cifar100":
        # Load raw numpy from CIFAR-100 pickle files directly (skip torchvision for train)
        raw_data, raw_labels = [], []
        for b in range(1, 6):
            with open(os.path.join(data_root, "cifar-100-python", f"data_batch_{b}"), "rb") as f:
                batch = pickle.load(f)
            raw_data.append(np.array(batch["data"]))
            raw_labels.append(np.array(batch["fine_labels"]))
        all_data = np.concatenate(raw_data).reshape(-1, 3, 32, 32) / 255.0
        all_images = torch.tensor(all_data, dtype=torch.float32)
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1,3,1,1)
        std  = torch.tensor([0.2675, 0.2565, 0.2761]).view(1,3,1,1)
        all_images = (all_images - mean) / std
        all_labels = torch.tensor(np.concatenate(raw_labels), dtype=torch.long)
        # Test set via torchvision
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        # Load test from pickle too (torchvision format incompatible)
        with open(os.path.join(data_root, "cifar-100-python", "test_batch"), "rb") as f:
            test_batch = pickle.load(f)
        test_data = np.array(test_batch["data"]).reshape(-1, 3, 32, 32) / 255.0
        test_imgs = torch.tensor(test_data, dtype=torch.float32)
        test_imgs = (test_imgs - mean) / std
        test_labels = torch.tensor(np.array(test_batch["fine_labels"]), dtype=torch.long)
        test_ds = TensorDataset(test_imgs, test_labels)
        N = len(all_labels)
        print(f"   Train: {N:,}  |  Test: {len(test_ds):,}  |  Classes: 100")
        return all_images, all_labels, N, DataLoader(test_ds, batch_size=64, shuffle=False), test_ds
        N = len(all_labels)
        classes = 100
        # Don't return train_ds for cifar100 (we used raw data instead)
        return all_images, all_labels, N, test_loader, test_ds
    else:
        # CIFAR-10: stack preprocessed tensors
        all_images = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
        all_labels = torch.tensor(train_ds.targets)
        N = len(train_ds)
    num_classes = 100 if name == "cifar100" else 10
    print(f"   Train: {N:,}  |  Test: {len(test_ds):,}  |  Classes: {num_classes}")
    return all_images, all_labels, N, test_loader, test_ds


# =============================================================================
# 3.  Single-experiment runner
# =============================================================================
def run_single_experiment(dataset_name, method, forget_ratio,
                           all_images, all_labels, N_total, test_loader,
                           device, cfg, output_dir, model_name="mlp",
                           finetune_variant="ascent+descent"):
    """
    Run one (method × forget_ratio) experiment.

    Args:
        model_name: 'mlp'/'mlp2'/mlp3'/'deepmlp'/'cnn'
        finetune_variant: 'ascent+descent' (safe) or 'ascent_only' (pure forgetting)
    
    Returns: dict of results.
    """
    bs = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]

    # ---- 3a.  Split retain / forget ----
    n_forget = int(N_total * forget_ratio)
    perm = torch.randperm(N_total, generator=torch.Generator().manual_seed(
        cfg["training"]["seed"] + int(forget_ratio * 1000)))
    forget_indices = perm[:n_forget].tolist()
    retain_indices = perm[n_forget:].tolist()

    def _loader(indices, shuf=False):
        idx = list(indices) if isinstance(indices, set) else indices
        imgs = all_images[idx]
        lbls = all_labels[idx]
        if imgs.dim() == 3 and dataset_name == "mnist":
            imgs = imgs.unsqueeze(1)
        ds = TensorDataset(imgs, lbls)
        return DataLoader(ds, batch_size=bs, shuffle=shuf)

    forget_loader = _loader(forget_indices, shuf=False)
    retain_loader = _loader(retain_indices, shuf=False)
    full_loader   = _loader(list(range(N_total)), shuf=True)

    # ---- 3b.  Train original model ----
    print(f"   🔧 Training original model...")
    t0 = time.time()
    model_orig = get_model(dataset_name, device, model_name)
    optimizer = torch.optim.Adam(model_orig.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model_orig.train()
    for ep in range(epochs):
        running = 0.0
        for im, lb in full_loader:
            im, lb = im.to(device), lb.to(device)
            optimizer.zero_grad()
            loss = criterion(model_orig(im), lb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        if ep % max(1, epochs // 5) == 0:
            print(f"      Epoch {ep+1}/{epochs}  loss={running/len(full_loader):.4f}")
    train_time = time.time() - t0
    print(f"      ✅  Trained in {train_time:.1f}s")

    # ---- 3c.  Evaluate original model (test accuracy) ----
    model_orig.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for im, lb in test_loader:
            im, lb = im.to(device), lb.to(device)
            preds = model_orig(im).argmax(dim=1)
            total += lb.size(0)
            correct += (preds == lb).sum().item()
    test_acc_orig = correct / total
    print(f"      Test accuracy (original): {test_acc_orig*100:.2f}%")

    # ---- 3d.  Apply unlearning ----
    is_sisa = False
    unlearn_model = None

    if method == "NoUnlearning":
        unlearn_model, unlearn_time = unlearn_none(model_orig)
        unlearn_model = clone_model(model_orig, dataset_name, device, model_name)

    elif method == "Retrain":
        unlearn_model, unlearn_time = unlearn_retrain(
            get_model, dataset_name, device,
            all_images, all_labels,
            retain_indices,
            epochs=epochs, lr=lr, batch_size=bs, model_name=model_name,
        )

    elif method == "SISA":
        sisa_cfg = cfg.get("sisa", {})
        unlearn_model, unlearn_time = unlearn_sisa(
            get_model, dataset_name, device,
            all_images, all_labels,
            forget_indices, retain_indices,
            N_total,
            sisa_num_shards=sisa_cfg.get("num_shards", 5),
            sisa_slices_per_shard=sisa_cfg.get("slices_per_shard", 10),
            sisa_epochs_per_slice=sisa_cfg.get("epochs_per_slice", 1),
            lr=lr, batch_size=bs, model_name=model_name,
        )
        is_sisa = True

    elif method == "FineTune":
        ft_cfg = cfg.get("finetune", {})
        train_retain = (finetune_variant == "ascent+descent")
        unlearn_model, unlearn_time = unlearn_finetune(
            model_orig, dataset_name, device,
            all_images, all_labels,
            forget_indices, retain_indices,
            finetune_epochs=ft_cfg.get("epochs", 5),
            finetune_lr=ft_cfg.get("learning_rate", 0.0001),
            finetune_train_on_retain=train_retain,
            model_name=model_name,
            batch_size=bs,
        )

    elif method == "RII":
        rii_cfg = cfg.get("rii", {})
        unlearn_model, unlearn_time = unlearn_rii_driven(
            model_orig, dataset_name, device,
            all_images, all_labels,
            forget_indices, retain_indices,
            rii_epochs=rii_cfg.get("epochs", 10),
            rii_lr=rii_cfg.get("lr", 0.0001),
            lambda_rii=rii_cfg.get("lambda", 1.0),
            rii_tau=rii_cfg.get("tau", 0.01),
            model_name=model_name,
            batch_size=bs,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"      ⏱  Unlearn time: {unlearn_time:.1f}s")

    # ---- 3e.  Evaluate unlearned model (test accuracy) ----
    if is_sisa:
        test_acc = _sisa_eval(unlearn_model, test_loader, device)
    else:
        unlearn_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for im, lb in test_loader:
                im, lb = im.to(device), lb.to(device)
                preds = unlearn_model(im).argmax(dim=1)
                total += lb.size(0)
                correct += (preds == lb).sum().item()
        test_acc = correct / total
    print(f"      Test accuracy (unlearned): {test_acc*100:.2f}%")

    # ---- 3f.  Information metrics ----
    info = compute_all_metrics(
        unlearn_model, forget_loader, retain_loader, device,
        n_forget=len(forget_indices), n_retain=len(retain_indices),
        is_sisa=is_sisa,
    )

    # ---- 3g.  MIA ----
    mia_results = run_mia(unlearn_model, forget_loader, retain_loader,
                           device, is_sisa=is_sisa)

    # ---- 3h.  Save model (optional) ----
    if not cfg.get("no_save_models", False) and not is_sisa:
        model_path = os.path.join(output_dir, "models",
                                   f"{dataset_name}_{method}_{forget_ratio:.2f}.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(unlearn_model.state_dict(), model_path)
    elif is_sisa and not cfg.get("no_save_models", False):
        sisa_dir = os.path.join(output_dir, "models",
                                 f"sisa_{dataset_name}_{forget_ratio:.2f}")
        os.makedirs(sisa_dir, exist_ok=True)
        for i, m in enumerate(unlearn_model):
            torch.save(m.state_dict(), os.path.join(sisa_dir, f"shard_{i}.pt"))

    # ---- 3i.  Assemble result row ----
    result = {
        "dataset":            dataset_name,
        "method":             method,
        "forget_ratio":       forget_ratio,
        "n_forget":           len(forget_indices),
        "n_retain":           len(retain_indices),
        "test_acc_original":  test_acc_orig,
        "test_acc_unlearned": test_acc,
        "train_time_sec":     train_time,
        "unlearn_time_sec":   unlearn_time,
        **info,
        **mia_results,
    }
    return result


def _sisa_eval(models, test_loader, device):
    """Aggregated evaluation for SISA (average softmax)."""
    for m in models:
        m.eval()
    correct, total = 0, 0
    with torch.no_grad():
        # Auto-detect num_classes from first model
        test_batch = next(iter(test_loader))[0][:1].to(device)
        nc = models[0](test_batch).size(1)
        for im, lb in test_loader:
            im, lb = im.to(device), lb.to(device)
            probs = torch.zeros(im.size(0), nc, device=device)
            for m in models:
                probs += torch.softmax(m(im), dim=1)
            probs /= len(models)
            preds = probs.argmax(dim=1)
            total += lb.size(0)
            correct += (preds == lb).sum().item()
    return correct / total


# =============================================================================
# 4.  Main
# =============================================================================
def main():
    args = parse_args()

    # ── Load config ──
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:
        print(f"⚠️  Config file '{args.config}' not found — using defaults.")
        cfg = {}

    # Merge CLI overrides into config
    cfg.setdefault("training", {})
    cfg["training"]["epochs"]     = args.epochs
    cfg["training"]["batch_size"] = args.batch_size
    cfg["training"]["learning_rate"] = args.lr
    cfg["training"]["seed"]       = args.seed
    cfg["no_save_models"]         = args.no_save
    cfg.setdefault("rii", {})
    cfg["rii"]["epochs"] = args.rii_epochs
    cfg["rii"]["lambda"] = args.lambda_rii
    cfg["rii"]["tau"]    = args.rii_tau

    # ── Set up ──
    set_seed(args.seed)
    device = get_device()
    print(f"   Device: {device}\n")

    # ── Output directory ──
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(cfg.get("output", {}).get("results_dir", "results"),
                               f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Output directory: {out_dir}\n")

    # ── Normalise & expand method names ──
    _name_map = {
        "none": "NoUnlearning", "nounlearning": "NoUnlearning",
        "retrain": "Retrain", "sisa": "SISA", "finetune": "FineTune",
        "rii": "RII",
        "NoUnlearning": "NoUnlearning", "Retrain": "Retrain",
        "SISA": "SISA", "FineTune": "FineTune", "RII": "RII",
    }
    methods = [_name_map.get(m.lower() if m != "all" else m, m)
               for m in args.unlearn_method]
    if "all" in methods:
        methods = ["NoUnlearning", "Retrain", "SISA", "FineTune", "RII"]

    # ── Load data ──
    print(f"📦 Loading {args.dataset.upper()} dataset...")
    all_images, all_labels, N_total, test_loader, _ = load_dataset(args.dataset)
    print()

    # ── Subset handling ──
    if args.subset > 0 and args.subset < N_total:
        print(f"📦 Using subset: {args.subset}/{N_total} training samples")
        perm_sub = torch.randperm(N_total, generator=torch.Generator().manual_seed(args.seed))
        subset_idx = perm_sub[:args.subset]
        all_images = all_images[subset_idx]
        all_labels = all_labels[subset_idx]
        N_total = args.subset
        print()

    # ── Run experiments ──
    all_results = []
    total_runs = args.num_seeds * len(methods) * len(args.forget_ratio)
    run_idx = 0

    for seed_offset in range(args.num_seeds):
        current_seed = args.seed + seed_offset
        set_seed(current_seed)

        for ratio in args.forget_ratio:
            for method in methods:
                run_idx += 1
                seed_tag = f" seed={current_seed}" if args.num_seeds > 1 else ""
                print(f"{'='*60}")
                print(f"  [{run_idx}/{total_runs}]  Method={method}  "
                      f"Forget Ratio={ratio*100:.0f}%  Dataset={args.dataset}"
                      f"  Model={args.model}{seed_tag}")
                print(f"{'='*60}")

                try:
                    result = run_single_experiment(
                        args.dataset, method, ratio,
                        all_images, all_labels, N_total, test_loader,
                        device, cfg, out_dir,
                        model_name=args.model,
                        finetune_variant=args.finetune_variant,
                    )
                    result["seed"] = current_seed
                    result["model"] = args.model
                    all_results.append(result)
                    print(f"   ✅  RII={result['rii_rho']:.6f}  "
                          f"MI_ub={result['mi_upper_bound']:.6f}  "
                          f"σ₂/σ₁={result['sigma_ratio']:.6f}  "
                          f"MIA_acc={result['mia_best_acc']*100:.1f}%\n")
                except Exception as e:
                    print(f"   ❌  FAILED: {e}\n")
                    import traceback
                    traceback.print_exc()

    # ── Save CSV ──
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📁 Results saved → {csv_path}")
    print(f"   {len(df)} rows, {len(df.columns)} columns\n")

    # ── Summary table ──
    if len(df) > 0:
        print("=" * 70)
        print("                    📋  SUMMARY")
        print("=" * 70)
        cols = ["method", "forget_ratio", "test_acc_unlearned", "unlearn_time_sec",
                "rii_rho", "mi_upper_bound", "sigma_ratio", "mia_best_acc"]
        avail_cols = [c for c in cols if c in df.columns]
        print(df[avail_cols].to_string(index=False))
        print("=" * 70)

        # ── Generate plots ──
        print("\n📊 Generating plots...")
        generate_all_plots(df, out_dir)
    else:
        print("\n⚠️  No successful experiments — skipping plots.")

    print(f"\n✅  Pipeline complete.  All outputs in: {out_dir}/")
    print(f"    CSV:    {csv_path}")
    print(f"    Plots:  {out_dir}/*.png")
    if not args.no_save:
        print(f"    Models: {out_dir}/models/")


if __name__ == "__main__":
    main()
