#!/usr/bin/env python3
"""
Machine Unlearning Experiment — Gold Standard Baseline (Retrain from Scratch)
=============================================================================

This script implements the "retrain from scratch" baseline for machine
unlearning on the MNIST dataset. The pipeline:

  1. Train a 2-layer neural network on the FULL MNIST training set.
  2. Split the training set into a "retain set" (90%) and a "forget set" (10%).
  3. Retrain the SAME model architecture from scratch on the RETAIN set only.
  4. Compare test-set accuracy and training times.

The retrained model serves as the gold-standard for "perfect unlearning":
the forget-set samples have zero influence on the final parameters.

Hardware: MacBook Pro (M5 Pro, 24 GB) — uses Apple Metal (MPS) acceleration.
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 0.  Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.mps.manual_seed(SEED)          # For MPS backend

# ---------------------------------------------------------------------------
# 1.  Device selection (MPS > CPU)
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅  Using Apple Metal (MPS) acceleration.")
elif torch.backends.mps.is_built():
    DEVICE = torch.device("cpu")
    print("⚠️  MPS is built but not available — check macOS >= 12.3. Falling back to CPU.")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  MPS not available on this hardware. Falling back to CPU.")

print(f"   Device: {DEVICE}")
print()

# ---------------------------------------------------------------------------
# 2.  Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE  = 64
EPOCHS      = 10
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128

RETAIN_RATIO = 0.90     # 90 % retain, 10 % forget

# ---------------------------------------------------------------------------
# 3.  Data loading — MNIST
# ---------------------------------------------------------------------------
print("📦 Loading MNIST dataset...")

transform = transforms.Compose([
    transforms.ToTensor(),                        # [0, 255] → [0.0, 1.0]
    transforms.Normalize((0.1307,), (0.3081,))    # MNIST mean & std
])

train_dataset_full = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"   Training samples : {len(train_dataset_full):,}")
print(f"   Test samples     : {len(test_dataset):,}")
print()

# ---------------------------------------------------------------------------
# 4.  Train / retain / forget split
# ---------------------------------------------------------------------------
retain_size = int(len(train_dataset_full) * RETAIN_RATIO)
forget_size = len(train_dataset_full) - retain_size

# We use a deterministic split (fixed seed) so results are reproducible.
retain_dataset, forget_dataset = random_split(
    train_dataset_full, [retain_size, forget_size],
    generator=torch.Generator().manual_seed(SEED)
)

print("✂️  Train-set split:")
print(f"   Retain set : {len(retain_dataset):,}  ({RETAIN_RATIO*100:.0f}%)")
print(f"   Forget set : {len(forget_dataset):,}  ({(1-RETAIN_RATIO)*100:.0f}%)")
print()

# Create DataLoaders
full_loader    = DataLoader(train_dataset_full, batch_size=BATCH_SIZE, shuffle=True)
retain_loader  = DataLoader(retain_dataset,      batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------------------------------------------------------
# 5.  Model definition — 2-layer MLP
# ---------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    """A minimal 2-layer feed-forward network for MNIST classification.

    Architecture:  Flatten(28×28) → Linear(784, H) → ReLU → Linear(H, 10)
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # Flatten: (N, 1, 28, 28) → (N, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_model() -> SimpleMLP:
    """Factory: creates a fresh (untrained) model on the chosen device."""
    return SimpleMLP(
        input_dim=784, hidden_dim=HIDDEN_SIZE, num_classes=10
    ).to(DEVICE)

# ---------------------------------------------------------------------------
# 6.  Training & evaluation helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    total_epochs: int,
) -> float:
    """Train for one epoch; returns average loss."""
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"   Epoch {epoch+1}/{total_epochs}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)


def train_model(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    label: str = "Model",
) -> tuple[nn.Module, float]:
    """Full training loop.  Returns (trained_model, elapsed_seconds)."""
    print(f"🚀 Training {label} ({epochs} epochs, lr={lr})...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    t_start = time.time()
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, loader, optimizer, criterion,
                                   epoch, epochs)
        print(f"   Epoch {epoch+1:2d}/{epochs}  |  avg loss = {avg_loss:.6f}")
    elapsed = time.time() - t_start

    print(f"   ✅  Done in {elapsed:.1f} s")
    print()
    return model, elapsed


def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    """Return classification accuracy on the given DataLoader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ---------------------------------------------------------------------------
# 7.  Experiment
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("   MACHINE UNLEARNING — Gold-Standard Baseline")
    print("   Retrain-from-scratch on the retain set")
    print("=" * 60)
    print()

    # --- 7a.  Train original model on FULL training set ---
    model_original = create_model()
    model_original, time_original = train_model(
        model_original, full_loader, label="Original (full train set)"
    )
    acc_original = evaluate_model(model_original, test_loader)

    # --- 7b.  Train retrained model on RETAIN set only ---
    model_retrained = create_model()              # <-- fresh random init
    model_retrained, time_retrained = train_model(
        model_retrained, retain_loader, label="Retrained (retain set only)"
    )
    acc_retrained = evaluate_model(model_retrained, test_loader)

    # --- 7c.  Report ---
    print("=" * 60)
    print("                         RESULTS")
    print("=" * 60)
    print(f"  {'Metric':<35} {'Original':>12} {'Retrained':>12}")
    print(f"  {'─'*35} {'─'*12} {'─'*12}")
    print(f"  {'Training data':<35} {'Full (60K)':>12} {'Retain (54K)':>12}")
    print(f"  {'Test accuracy':<35} {acc_original*100:>11.2f}% {acc_retrained*100:>11.2f}%")
    print(f"  {'Training time':<35} {time_original:>10.1f}s  {time_retrained:>10.1f}s ")
    print()
    print(f"  Δ Accuracy  (retrained − original) : {(acc_retrained - acc_original)*100:+.2f}%")
    print(f"  Δ Time      (retrained − original) : {time_retrained - time_original:+.1f}s")
    print("=" * 60)

    # -------------------------------------------------------------------
    # 8.  (Optional) Quick sanity check on the forget set
    # -------------------------------------------------------------------
    forget_loader = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=False)
    acc_original_forget = evaluate_model(model_original,  forget_loader)
    acc_retrained_forget = evaluate_model(model_retrained, forget_loader)

    print()
    print("  ── Forget-set sanity check ──")
    print(f"  Original  accuracy on forget set : {acc_original_forget*100:.2f}%")
    print(f"  Retrained accuracy on forget set : {acc_retrained_forget*100:.2f}%")
    print(f"  (Retrained model was never trained on these samples.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
