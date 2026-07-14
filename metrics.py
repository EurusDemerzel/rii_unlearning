"""
rii_unlearning — Information-theoretic metrics for evaluating machine unlearning quality.

Metrics (based on the rank-one channel model):
  1. Symmetric KL divergence between P(Y|forget) and P(Y|retain)
  2. Mutual information I(X;Y) where X = forget/retain indicator, Y = prediction
  3. Channel rank metric: σ₂/σ₁ ratio of the 2×K channel matrix
     (σ₂/σ₁ → 0 means rank-1 = perfect unlearning)
"""

import numpy as np
import torch


def extract_predictions(model, loader, device, is_sisa=False, num_classes=100):
    """
    Extract softmax probabilities and hard predictions from a model.

    Args:
        model:   single nn.Module OR list of nn.Module (for SISA aggregation)
        loader:  DataLoader
        device:  torch device
        is_sisa: if True, `model` is a list of shard models → average softmax

    Returns:
        probs_np: (N, K) softmax probabilities
        preds_np: (N,)   hard argmax predictions
    """
    if is_sisa:
        for m in model:
            m.eval()
    else:
        model.eval()

    all_probs, all_preds = [], []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            if is_sisa:
                # Auto-detect num_classes from first model's output
                test_out = model[0](images[:1])
                nc = test_out.size(1)
                probs = torch.zeros(images.size(0), nc, device=device)
                for m in model:
                    probs += torch.softmax(m(images), dim=1)
                probs /= len(model)
            else:
                logits = model(images)
                probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_preds, axis=0)


def compute_kl_divergence(probs_forget, probs_retain, eps=1e-10):
    """
    KL divergence between average softmax distributions on forget vs retain sets.

    Returns:
        kl_forward:   KL(P_forget_avg || P_retain_avg)
        kl_backward:  KL(P_retain_avg || P_forget_avg)
        kl_symmetric: (kl_forward + kl_backward) / 2
    """
    p_f = np.mean(probs_forget, axis=0) + eps
    p_r = np.mean(probs_retain, axis=0) + eps
    p_f /= p_f.sum()
    p_r /= p_r.sum()

    kl_fwd = np.sum(p_f * np.log(p_f / p_r))
    kl_bwd = np.sum(p_r * np.log(p_r / p_f))
    return kl_fwd, kl_bwd, (kl_fwd + kl_bwd) / 2.0


def compute_mutual_information(preds_forget, preds_retain,
                                n_forget, n_retain, num_classes=100, eps=1e-10):
    """
    Estimate I(X;Y) where X ∈ {forget, retain} and Y is the predicted class.
    Uses empirical frequency distributions.

    I(X;Y) = H(Y) − H(Y|X)
    """
    # P(Y | X=forget)
    py_gf = np.bincount(preds_forget, minlength=num_classes).astype(float)
    py_gf /= max(py_gf.sum(), 1)

    # P(Y | X=retain)
    py_gr = np.bincount(preds_retain, minlength=num_classes).astype(float)
    py_gr /= max(py_gr.sum(), 1)

    total = n_forget + n_retain
    px_f = n_forget / total
    px_r = n_retain / total

    # P(Y) mixture
    py = px_f * py_gf + px_r * py_gr

    h_y  = -np.sum(py * np.log(py + eps))
    h_yx = px_f * (-np.sum(py_gf * np.log(py_gf + eps))) + \
           px_r * (-np.sum(py_gr * np.log(py_gr + eps)))

    return max(h_y - h_yx, 0.0)


def compute_channel_rank(probs_forget, probs_retain):
    """
    Compute the effective rank of the 2×K channel matrix P(Y|X).

    Row 0: P(Y | X=retain)
    Row 1: P(Y | X=forget)

    Returns:
        sigma_ratio:     σ₂ / σ₁  (→0 means rank-1)
        singular_values: [σ₁, σ₂]
    """
    p_r = np.mean(probs_retain, axis=0)
    p_f = np.mean(probs_forget, axis=0)
    channel = np.stack([p_r, p_f], axis=0)          # (2, K)
    _, s, _ = np.linalg.svd(channel, full_matrices=False)
    ratio = s[1] / s[0] if s[0] > 1e-12 else 0.0
    return ratio, s


def compute_all_metrics(model, forget_loader, retain_loader, device,
                        n_forget, n_retain, is_sisa=False):
    """
    Convenience: run all information metrics at once.

    Returns dict with keys:
      kl_symmetric, kl_forward, kl_backward,
      mutual_information, sigma_ratio, sigma_1, sigma_2,
      rii_rho, mi_upper_bound
    """
    probs_f, preds_f = extract_predictions(model, forget_loader, device, is_sisa)
    probs_r, preds_r = extract_predictions(model, retain_loader, device, is_sisa)

    kl_fwd, kl_bwd, kl_sym = compute_kl_divergence(probs_f, probs_r)
    mi = compute_mutual_information(preds_f, preds_r, n_forget, n_retain)
    sr, sv = compute_channel_rank(probs_f, probs_r)
    rho, mi_ub = compute_rii_from_probs(probs_f, probs_r)

    return {
        "kl_symmetric":       kl_sym,
        "kl_forward":         kl_fwd,
        "kl_backward":        kl_bwd,
        "mutual_information": mi,
        "sigma_ratio":        sr,
        "sigma_1":            sv[0],
        "sigma_2":            sv[1],
        "rii_rho":            rho,
        "mi_upper_bound":     mi_ub,
    }


def compute_rii_from_probs(probs_forget, probs_retain, eps=1e-12):
    """
    Compute the Normalised Residual Information Index (RII) ρ and the
    tight mutual-information upper bound from averaged softmax outputs.

    RII:  ρ = 1 − σ₁² / (σ₁² + σ₂²) ∈ [0, 0.5]
         ρ = 0 ⇔ rank-1 (perfect unlearning)
         ρ → 0.5 ⇔ maximal distinguishability

    MI upper bound:  I ≤ ½ log₂(1 / (1 − 2ρ))  (nats)

    Args:
        probs_forget: (N_f, K) softmax probabilities on forget set
        probs_retain: (N_r, K) softmax probabilities on retain set

    Returns:
        rho:    Residual Information Index (float)
        mi_ub:  tight mutual-information upper bound in nats (float)
    """
    import numpy as np

    mu_f = np.mean(probs_forget, axis=0).reshape(1, -1)
    mu_r = np.mean(probs_retain, axis=0).reshape(1, -1)
    M = np.vstack([mu_f, mu_r])                     # 2 × K empirical confusion matrix

    _, S, _ = np.linalg.svd(M, full_matrices=False)
    s1, s2 = S[0], S[1]

    rho = 1.0 - s1**2 / (s1**2 + s2**2 + eps)
    mi_ub = 0.5 * np.log(1.0 / max(1.0 - 2.0 * rho, eps))

    return float(rho), float(mi_ub)
