"""
Unlearning methods for the machine unlearning pipeline.

Implements:
  - NoUnlearning  (baseline — original model, no forgetting)
  - Retrain       (gold standard — retrain from scratch on retain set)
  - SISA          (sharded, isolated, sliced, aggregated — Bourtoule et al. 2021)
  - FineTune      (gradient-ascent on forget set)
"""

import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import get_model, clone_model

# =============================================================================
# Shared helpers
# =============================================================================
def _indices_to_loader(images, labels, indices, batch_size=64, shuffle=True):
    """Convert global indices to a DataLoader."""
    idx_list = list(indices) if isinstance(indices, set) else indices
    imgs = images[idx_list]
    lbls = labels[idx_list]
    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(1)            # MNIST: (N, 28, 28) → (N, 1, 28, 28)
    ds = TensorDataset(imgs, lbls)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _train_epochs(model, loader, epochs, lr, device, desc="", quiet=True):
    """Train model for `epochs` on `loader`.  Returns trained model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    iterator = range(epochs) if quiet else tqdm(range(epochs), desc=desc, leave=False)
    for _ in iterator:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    return model


# =============================================================================
# 1.  No Unlearning (baseline)
# =============================================================================
def unlearn_none(model, **kwargs):
    """Return the original model unchanged.  Time = 0."""
    return model, 0.0


# =============================================================================
# 2.  Retrain from Scratch (gold standard)
# =============================================================================
def unlearn_retrain(model_class_fn, dataset_name, device,
                    all_images, all_labels,
                    retain_indices, **kwargs):
    """
    Train a brand-new model only on the retain set.

    Returns: (unlearned_model, time_seconds)
    """
    epochs = kwargs.get("epochs", 10)
    lr     = kwargs.get("lr", 0.001)
    bs     = kwargs.get("batch_size", 64)
    model_name = kwargs.get("model_name", "mlp")

    t0 = time.time()
    model = model_class_fn(dataset_name, device, model_name)
    loader = _indices_to_loader(all_images, all_labels, retain_indices,
                                batch_size=bs, shuffle=True)
    model = _train_epochs(model, loader, epochs=epochs, lr=lr, device=device,
                          desc="Retrain")
    return model, time.time() - t0


# =============================================================================
# 3.  SISA  (Sharded, Isolated, Sliced, Aggregated)
# =============================================================================
def unlearn_sisa(model_class_fn, dataset_name, device,
                 all_images, all_labels,
                 forget_indices, retain_indices,
                 N_total, **kwargs):
    """
    SISA unlearning (Bourtoule et al., IEEE S&P 2021).

    Steps:
      1. Partition all data into S shards.
      2. Within each shard, split into T slices.
      3. Train each shard incrementally (slice-by-slice, saving checkpoints).
      4. On unlearning request: rewind affected shards to the last clean
         checkpoint, then retrain only the remaining clean slices.

    Returns: (list_of_shard_models, unlearn_time_seconds)
    """
    S = kwargs.get("sisa_num_shards", 5)
    T = kwargs.get("sisa_slices_per_shard", 10)
    eps = kwargs.get("sisa_epochs_per_slice", 1)
    lr  = kwargs.get("lr", 0.001)
    bs  = kwargs.get("batch_size", 64)
    model_name = kwargs.get("model_name", "mlp")

    # ---- 3a.  Shuffle & partition into S shards ----
    perm = torch.randperm(N_total)
    shard_size = N_total // S
    shard_indices = []
    for s in range(S):
        start = s * shard_size
        end   = start + shard_size if s < S - 1 else N_total
        shard_indices.append(perm[start:end])

    # ---- 3b.  Each shard → T slices ----
    slice_indices = []
    for s in range(S):
        s_idx = shard_indices[s]
        s_len = len(s_idx)
        sl_size = s_len // T
        slices = []
        for t in range(T):
            start = t * sl_size
            end   = start + sl_size if t < T - 1 else s_len
            slices.append(s_idx[start:end])
        slice_indices.append(slices)

    # ---- 3c.  Identify affected shards & first affected slice ----
    forget_set = set(forget_indices)
    shard_first_affected = [T] * S
    shard_has_forget = [False] * S

    for s in range(S):
        for t in range(T):
            if set(slice_indices[s][t].tolist()) & forget_set:
                shard_has_forget[s] = True
                if t < shard_first_affected[s]:
                    shard_first_affected[s] = t

    # ---- 3d.  Incremental training per shard (with checkpoints) ----
    all_checkpoints = []                # [s][t] → model after slice t
    for s in range(S):
        checkpoints = []
        model = model_class_fn(dataset_name, device, model_name)
        cumulative = []
        for t in range(T):
            cumulative.extend(slice_indices[s][t].tolist())
            loader = _indices_to_loader(all_images, all_labels, cumulative,
                                        batch_size=bs, shuffle=True)
            model = _train_epochs(model, loader, epochs=eps, lr=lr, device=device)
            checkpoints.append(clone_model(model, dataset_name, device, model_name))
        all_checkpoints.append(checkpoints)

    # ---- 3e.  SISA unlearning — rewind & retrain affected shards ----
    t0 = time.time()
    unlearned_models = []

    for s in range(S):
        if not shard_has_forget[s]:
            unlearned_models.append(clone_model(all_checkpoints[s][-1], dataset_name, device, model_name,
                                                 dataset_name, device))
            continue

        k = shard_first_affected[s]
        if k == 0:
            # No clean checkpoint → retrain entire shard from scratch
            clean = []
            for t in range(T):
                sl = slice_indices[s][t].tolist()
                clean.extend([i for i in sl if i not in forget_set])
            model = model_class_fn(dataset_name, device, model_name)
            if clean:
                loader = _indices_to_loader(all_images, all_labels, clean,
                                            batch_size=bs, shuffle=True)
                model = _train_epochs(model, loader, epochs=eps * T, lr=lr,
                                      device=device)
            unlearned_models.append(model)
        else:
            # Rewind to checkpoint k-1, retrain on clean data from slices k..T-1
            model = clone_model(all_checkpoints[s][k - 1], dataset_name, device, model_name)
            clean = []
            for t in range(k, T):
                sl = slice_indices[s][t].tolist()
                clean.extend([i for i in sl if i not in forget_set])
            remaining_epochs = eps * (T - k)
            if clean:
                loader = _indices_to_loader(all_images, all_labels, clean,
                                            batch_size=bs, shuffle=True)
                model = _train_epochs(model, loader, epochs=remaining_epochs,
                                      lr=lr, device=device)
            unlearned_models.append(model)

    elapsed = time.time() - t0

    # Return the list of shard models (pipeline.py handles aggregation)
    return unlearned_models, elapsed


# =============================================================================
# 4.  Fine-Tune Unlearning (gradient ascent on forget set)
# =============================================================================
def unlearn_finetune(model, dataset_name, device,
                     all_images, all_labels,
                     forget_indices, retain_indices, **kwargs):
    """
    Fine-tune unlearning: run gradient *ascent* on the forget set so the
    model maximises loss on data it should forget.

    Optionally simultaneously train normally on the retain set to preserve
    overall accuracy (controlled by `train_on_retain` flag).

    Returns: (unlearned_model, time_seconds)
    """
    epochs       = kwargs.get("finetune_epochs", 5)
    lr           = kwargs.get("finetune_lr", 0.0001)
    bs           = kwargs.get("batch_size", 64)
    train_retain = kwargs.get("finetune_train_on_retain", False)
    model_name   = kwargs.get("model_name", "mlp")

    model = clone_model(model, dataset_name, device, model_name)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    forget_loader = _indices_to_loader(all_images, all_labels, forget_indices,
                                        batch_size=bs, shuffle=True)
    retain_loader = _indices_to_loader(all_images, all_labels, retain_indices,
                                        batch_size=bs, shuffle=True) if train_retain else None

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        # Gradient ascent on forget set
        for images, labels in forget_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            (-loss).backward()                              # negate → ascent
            optimizer.step()

        # Optional: normal training on retain set (preserves accuracy)
        if retain_loader is not None:
            for images, labels in retain_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

    return model, time.time() - t0

# =============================================================================
# 5.  RII-Driven Unlearning (RII as direct optimization target)
# =============================================================================
def unlearn_rii_driven(model, dataset_name, device,
                        all_images, all_labels,
                        forget_indices, retain_indices, **kwargs):
    """
    RII-driven unlearning: directly minimize ||mu_f - mu_r||^2 as a proxy for RII.

    This avoids SVD differentiation by optimizing the distance between the
    average softmax outputs on the forget and retain sets.

    Loss = CrossEntropy(retain) + lambda_rii * ||mu_f - mu_r||^2

    Early stopping: stop when ||mu_f - mu_r|| < tau (empirically, this
    corresponds to rho < threshold).

    Returns: (unlearned_model, time_seconds)
    """
    epochs       = kwargs.get("rii_epochs", 10)
    lr           = kwargs.get("rii_lr", 0.0001)
    bs           = kwargs.get("batch_size", 64)
    lambda_rii   = kwargs.get("lambda_rii", 1.0)
    tau          = kwargs.get("rii_tau", 0.01)     # early-stop threshold

    model = clone_model(model, dataset_name, device, 
                         kwargs.get("model_name", "mlp"))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    forget_loader = _indices_to_loader(all_images, all_labels, forget_indices,
                                        batch_size=bs, shuffle=True)
    retain_loader = _indices_to_loader(all_images, all_labels, retain_indices,
                                        batch_size=bs, shuffle=True)

    def _avg_softmax(loader):
        """Compute mean softmax over a loader."""
        model.eval()
        all_probs = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                all_probs.append(torch.softmax(model(images), dim=1))
        return torch.cat(all_probs, dim=0).mean(dim=0)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for (f_imgs, f_lbls), (r_imgs, r_lbls) in zip(forget_loader, retain_loader):
            f_imgs, f_lbls = f_imgs.to(device), f_lbls.to(device)
            r_imgs, r_lbls = r_imgs.to(device), r_lbls.to(device)

            # Task loss on retain set
            retain_loss = criterion(model(r_imgs), r_lbls)

            # RII proxy: distance between mean predictions
            f_probs = torch.softmax(model(f_imgs), dim=1)
            r_probs = torch.softmax(model(r_imgs), dim=1)
            rii_proxy = torch.norm(f_probs.mean(dim=0) - r_probs.mean(dim=0), p=2)

            loss = retain_loss + lambda_rii * rii_proxy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Early stopping check
        mu_f = _avg_softmax(forget_loader)
        mu_r = _avg_softmax(retain_loader)
        dist = torch.norm(mu_f - mu_r, p=2).item()
        if dist < tau:
            break

    return model, time.time() - t0
