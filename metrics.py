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


# ═══════════════════════════════════════════════════════════════
# Per-Sample State Disparity (PSSD) functions
# ═══════════════════════════════════════════════════════════════

def quantize_logits(logits, n_clusters=20, random_state=42, batch_size=1024):
    """
    Fit k-means on logits and return state labels + kmeans model.

    Args:
        logits: (N, C) array of logit vectors
        n_clusters: number of states M (default 20, matching 2×C for C=10)

    Returns:
        states: (N,) integer state labels
        kmeans: trained MiniBatchKMeans model
    """
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state,
                              batch_size=batch_size)
    states = kmeans.fit_predict(logits)
    return states, kmeans


def compute_state_distribution(states, num_states, eps=1e-12):
    """Compute empirical state distribution from state labels."""
    hist = np.bincount(states, minlength=num_states).astype(float)
    return hist / (hist.sum() + eps)


def compute_individual_state_disparity(states_forget, states_retain, num_states):
    """
    Compute Individual State Disparity (ISD) δ(x) for each forget sample.

    δ(x) = 1 - π_r(s(x))  where π_r is the retain state distribution.

    Returns:
        disparities: (N_f,) array of δ(x) ∈ [0, 1-1/M]
        pi_r: (M,) retain state distribution
        delta_f: mean per-sample disparity Δ_f
        delta_r: mean retain disparity Δ_r (baseline self-surprise)
        excess: Ψ = Δ_f - Δ_r
    """
    pi_r = compute_state_distribution(states_retain, num_states)
    # Per-sample disparity for forget samples
    disparities = 1.0 - pi_r[states_forget]
    delta_f = float(np.mean(disparities))
    # Per-sample disparity for retain samples (baseline)
    disparities_r = 1.0 - pi_r[states_retain]
    delta_r = float(np.mean(disparities_r))
    excess = delta_f - delta_r
    return disparities, pi_r, delta_f, delta_r, excess


def compute_top_k_disparity(disparities, k=5):
    """Top-k disparity: mean of the k highest δ(x) values."""
    sorted_d = np.sort(disparities)[::-1]
    k = min(k, len(sorted_d))
    return float(np.mean(sorted_d[:k])) if k > 0 else 0.0


def compute_ps_mhpr(states_forget, states_heldout_list, num_states):
    """
    Per-sample MHPR: for each forget sample, project its one-hot state vector
    onto the convex hull of held-out state distributions.

    Returns:
        per_sample_residuals: (N_f,) array of per-sample residuals
        rho_H_ps: mean per-sample MHPR
        rho_H_std: standard MHPR (on averaged distributions)
    """
    # Standard MHPR (class-level average)
    pi_f = compute_state_distribution(states_forget, num_states)
    H = np.array([compute_state_distribution(s, num_states)
                  for s in states_heldout_list])
    K = H.shape[0]
    H_Ht = H @ H.T
    H_Ht_inv = np.linalg.inv(H_Ht + 1e-12 * np.eye(K))
    alpha = H_Ht_inv @ (H @ pi_f)
    pi_f_proj = H.T @ alpha
    residual_std = pi_f - pi_f_proj
    rho_H_std = float(np.sum(residual_std**2) / (np.sum(pi_f**2) + 1e-12))

    # Per-sample: each sample's one-hot projected
    K = H.shape[0]
    H_Ht_inv = np.linalg.inv(H @ H.T + 1e-12 * np.eye(K))
    per_sample_residuals = []
    for s in states_forget:
        e_s = np.zeros(num_states)
        e_s[s] = 1.0
        alpha_s = H_Ht_inv @ (H @ e_s)
        e_proj = H.T @ alpha_s
        res_sq = np.sum((e_s - e_proj)**2)
        per_sample_residuals.append(res_sq / (np.sum(e_s**2) + 1e-12))
    per_sample_residuals = np.array(per_sample_residuals)
    rho_H_ps = float(np.mean(per_sample_residuals))
    return per_sample_residuals, rho_H_ps, rho_H_std


def compute_ps_mia_roc(disparities_forget, disparities_retain):
    """
    ROC curve for MIA using per-sample disparity δ(x) as feature.
    Threshold τ: predict 'forget' if δ(x) ≥ τ.

    Returns:
        fpr, tpr, thresholds, auc
    """
    from sklearn.metrics import roc_curve, auc
    y_true = np.concatenate([np.ones_like(disparities_forget),
                             np.zeros_like(disparities_retain)])
    y_score = np.concatenate([disparities_forget, disparities_retain])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_score


def compute_ps_metrics_from_logits(logits_forget, logits_retain,
                                    logits_heldout_list=None,
                                    n_clusters=20, top_k=5):
    """
    End-to-end PSSD computation from logit arrays.

    Args:
        logits_forget: (N_f, C) logits of forget class samples
        logits_retain: (N_r, C) logits of retain class samples
        logits_heldout_list: list of (N_hk, C) logits for K held-out classes
        n_clusters: number of states M
        top_k: k for top-k disparity

    Returns:
        dict with all PSSD metrics
    """
    # Pool logits for global k-means
    all_logits = np.concatenate([logits_forget, logits_retain] +
                                (logits_heldout_list if logits_heldout_list else []))
    states_all, kmeans = quantize_logits(all_logits, n_clusters=n_clusters)

    # Split states back
    n_f = len(logits_forget)
    n_r = len(logits_retain)
    states_forget = states_all[:n_f]
    states_retain = states_all[n_f:n_f+n_r]

    # ISD
    disparities, pi_r, delta_f, delta_r, excess = \
        compute_individual_state_disparity(states_forget, states_retain, n_clusters)

    # Top-k
    top_k_val = compute_top_k_disparity(disparities, k=top_k)

    # Standard state-space RII
    pi_f = compute_state_distribution(states_forget, n_clusters)
    M_ps = np.vstack([pi_f.reshape(1, -1), pi_r.reshape(1, -1)])
    _, S, _ = np.linalg.svd(M_ps, full_matrices=False)
    rho_S = float(S[1]**2 / (S[0]**2 + S[1]**2 + 1e-12))

    results = {
        'delta_f': delta_f,
        'delta_r': delta_r,
        'excess': excess,
        'top_k_disparity': top_k_val,
        'rho_S': rho_S,
        'num_states': n_clusters,
        'n_f': n_f,
        'n_r': n_r,
    }

    # MIA ROC
    try:
        fpr, tpr, _, auc_val = compute_ps_mia_roc(disparities, 1.0 - pi_r[states_retain])
        results['mia_auc'] = auc_val
    except Exception:
        results['mia_auc'] = -1.0

    # Per-sample MHPR if heldout provided
    if logits_heldout_list:
        states_heldout = []
        offset = n_f + n_r
        for i, lh in enumerate(logits_heldout_list):
            states_heldout.append(states_all[offset:offset+len(lh)])
            offset += len(lh)
        _, rho_H_ps, rho_H_std = compute_ps_mhpr(states_forget, states_heldout, n_clusters)
        results['rho_H_ps'] = rho_H_ps
        results['rho_H_std'] = rho_H_std

    return results


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
