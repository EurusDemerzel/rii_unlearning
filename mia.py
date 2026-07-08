"""
Membership Inference Attack (MIA) for evaluating machine unlearning.

After unlearning, a successful MIA should NOT be able to distinguish
forget-set samples from retain-set samples (AUC ≈ 0.5, accuracy ≈ 50 %).

Implementation:  loss-based threshold attack.
  - Higher per-sample loss → more likely to be in the forget set
  - Lower per-sample loss → more likely to be in the retain set
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score


def _compute_per_sample_loss(model, loader, device, is_sisa=False):
    """
    Compute per-sample cross-entropy loss.

    Returns: losses (N,) numpy array
    """
    if is_sisa:
        for m in model:
            m.eval()
    else:
        model.eval()

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_sisa:
                # Aggregate logits by averaging
                logits = torch.zeros(images.size(0), num_classes, device=device)
                for m in model:
                    logits += m(images)
                logits /= len(model)
            else:
                logits = model(images)

            loss = criterion(logits, labels)
            all_losses.append(loss.cpu().numpy())

    return np.concatenate(all_losses, axis=0)


def run_mia(model, forget_loader, retain_loader, device, is_sisa=False):
    """
    Run a loss-based membership inference attack.

    Labels:  1 = forget-set member, 0 = retain-set member.
    Score:   per-sample cross-entropy loss (higher → predict forget).

    Returns dict with keys:
        mia_auc:          ROC AUC
        mia_best_acc:     best attack accuracy at optimal threshold
        mia_random_acc:   0.5 (random-guessing baseline)
        mia_forget_mean_loss:  average loss on forget set
        mia_retain_mean_loss:  average loss on retain set
    """
    losses_forget = _compute_per_sample_loss(model, forget_loader, device, is_sisa)
    losses_retain = _compute_per_sample_loss(model, retain_loader, device, is_sisa)

    # Build dataset for MIA
    scores  = np.concatenate([losses_forget, losses_retain])
    labels  = np.concatenate([np.ones(len(losses_forget)),
                               np.zeros(len(losses_retain))])

    # ROC AUC
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5

    # Best threshold accuracy (simple sweep)
    best_acc = 0.5
    for thresh in np.percentile(scores, np.linspace(5, 95, 50)):
        preds = (scores >= thresh).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc

    return {
        "mia_auc":              auc,
        "mia_best_acc":         best_acc,
        "mia_random_baseline":  0.5,
        "mia_forget_mean_loss": float(np.mean(losses_forget)),
        "mia_retain_mean_loss": float(np.mean(losses_retain)),
    }
