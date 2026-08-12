#!/usr/bin/env python3
"""
Visualization script for Per-Sample State Disparity (PSSD) experiments.

Generates 5 figures:
  Fig 1: Per-sample disparity histograms (forget vs retain)
  Fig 2: PSSD vs RII scatter plot across forgetting strengths
  Fig 3: ROC curves (δ-based MIA vs softmax-based MIA)
  Fig 4: Top-k disparity as function of gradient steps
  Fig 5: Per-sample MHPR vs standard MHPR (Jensen gap)
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Paths
RESULTS_DIR = "results/ps_experiment"
OUTPUT_DIR = "results/ps_experiment/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme
C_FORGET = '#E74C3C'
C_RETAIN = '#3498DB'
C_PSSD = '#2ECC71'
C_RII = '#9B59B6'
C_MIA_PS = '#E67E22'
C_MIA_SM = '#95A5A6'


def load_results(dataset, M=20, K=3):
    """Load PSSD experiment results."""
    path = os.path.join(RESULTS_DIR, dataset, f"ps_M{M}_K{K}.json")
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def compute_roc_from_disparities(states_f, states_r, num_states_M, disparities_f):
    """Helper: compute ROC from per-sample disparities."""
    pass  # ROC computation is in run_ps_experiment, we load from results


def fig1_disparity_histograms(datasets, M=20):
    """Figure 1: Distribution of δ(x) for forget vs retain."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        results = load_results(ds, M)
        if results is None:
            ax.set_title(f"{ds}: no data")
            continue

        # Use the midpoint of forget steps (where leakage is moderate)
        mid_idx = len(results) // 2
        r = results[mid_idx]

        # We need per-sample disparities; these aren't saved in the JSON.
        # Instead, show the aggregate metrics across steps
        steps = [res['steps'] for res in results]
        delta_f = [res['delta_f'] for res in results]
        delta_r = [res['delta_r'] for res in results]
        excess = [res['excess'] for res in results]

        ax.plot(steps, delta_f, 'o-', color=C_FORGET, label=r'$\Delta_f$ (forget)')
        ax.plot(steps, delta_r, 's-', color=C_RETAIN, label=r'$\Delta_r$ (retain)')
        ax.fill_between(steps,
                        [d - e for d, e in zip(delta_f, excess)],
                        delta_f, alpha=0.2, color=C_FORGET,
                        label=r'$\Psi$ (excess)')
        ax.set_xlabel('Gradient ascent steps')
        ax.set_ylabel('Mean disparity')
        ax.set_title(f'{ds.upper()} (M={M})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'fig1_disparity_histograms_M{M}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def fig2_pssd_vs_rii(datasets, M=20):
    """Figure 2: PSSD metrics vs standard RII scatter."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        results = load_results(ds, M)
        if results is None:
            ax.set_title(f"{ds}: no data")
            continue

        rho_sm = [res['rho_sm'] for res in results]
        delta_f = [res['delta_f'] for res in results]
        excess = [res['excess'] for res in results]
        steps = [res['steps'] for res in results]

        # Normalize for arrow labels
        sc = ax.scatter(rho_sm, delta_f, c=steps, cmap='viridis',
                        s=80, edgecolors='k', linewidths=0.5, zorder=5)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Gradient steps')

        # Add step annotations
        for i, s in enumerate(steps):
            if i % 2 == 0 or i == len(steps) - 1:
                ax.annotate(str(s), (rho_sm[i], delta_f[i]),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7)

        # Diagonal
        max_val = max(max(rho_sm), max(delta_f))
        ax.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.5,
                label='y=x')

        ax.set_xlabel(r'Softmax RII $\rho_{\mathrm{sm}}$')
        ax.set_ylabel(r'MPSD $\Delta_f$')
        ax.set_title(f'{ds.upper()} (M={M})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'fig2_pssd_vs_rii_M{M}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def fig3_roc_curves(datasets, M=20):
    """Figure 3: ROC curves for δ-based vs softmax-based MIA."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        results = load_results(ds, M)
        if results is None:
            ax.set_title(f"{ds}: no data")
            continue

        for i, res in enumerate(results):
            steps = res['steps']
            # We can't reconstruct full ROC from saved metrics.
            # Plot AUC vs steps instead
            pass

        # Plot AUC vs steps for both MIA types
        steps = [res['steps'] for res in results]
        auc_ps = [res['mia_auc'] for res in results]
        auc_sm = [res['mia_auc_sm'] for res in results]

        ax.plot(steps, auc_ps, 'o-', color=C_MIA_PS, label=r'δ-based MIA')
        ax.plot(steps, auc_sm, 's--', color=C_MIA_SM, label=r'Softmax MIA')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
        ax.set_xlabel('Gradient ascent steps')
        ax.set_ylabel('AUC')
        ax.set_title(f'{ds.upper()} (M={M})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.05)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'fig3_mia_auc_M{M}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def fig4_topk_disparity(datasets, M=20):
    """Figure 4: Top-k disparity vs gradient steps."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        results = load_results(ds, M)
        if results is None:
            ax.set_title(f"{ds}: no data")
            continue

        steps = [res['steps'] for res in results]
        delta_f = [res['delta_f'] for res in results]
        top5 = [res['top_k_5'] for res in results]
        top10 = [res['top_k_10'] for res in results]
        top20 = [res['top_k_20'] for res in results]

        ax.plot(steps, delta_f, 'o-', color=C_FORGET, label=r'$\Delta_f$ (mean)')
        ax.plot(steps, top5, 's-', color='red', label='Top-5')
        ax.plot(steps, top10, 'd-', color='orange', label='Top-10')
        ax.plot(steps, top20, '^--', color='goldenrod', alpha=0.7, label='Top-20')

        ax.set_xlabel('Gradient ascent steps')
        ax.set_ylabel('Disparity')
        ax.set_title(f'{ds.upper()} (M={M})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'fig4_topk_disparity_M{M}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def fig5_ps_mhpr_jensen(datasets, M=20):
    """Figure 5: Per-sample MHPR vs standard MHPR (Jensen gap)."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        results = load_results(ds, M)
        if results is None:
            ax.set_title(f"{ds}: no data")
            continue

        # Check if MHPR metrics exist
        if 'rho_H_ps' not in results[0]:
            ax.set_title(f"{ds}: no MHPR data")
            continue

        steps = [res['steps'] for res in results]
        rho_H_std = [res['rho_H_std'] for res in results]
        rho_H_ps = [res['rho_H_ps'] for res in results]
        gap = [ps - std for ps, std in zip(rho_H_ps, rho_H_std)]

        ax.plot(steps, rho_H_std, 'o-', color=C_RII, label=r'$\rho_{H,\mathcal{S}}$ (std)')
        ax.plot(steps, rho_H_ps, 's-', color=C_PSSD, label=r'$\rho_{H,\mathcal{S}}$ (per-sample)')
        ax.fill_between(steps, rho_H_std, rho_H_ps, alpha=0.2, color='gray',
                        label='Jensen gap')

        ax.set_xlabel('Gradient ascent steps')
        ax.set_ylabel(r'$\rho_H$')
        ax.set_title(f'{ds.upper()} (M={M})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'fig5_ps_mhpr_M{M}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def generate_all_figures(datasets=None, M=20):
    """Generate all 5 figures."""
    if datasets is None:
        # Find available datasets
        datasets = []
        for d in ['mnist', 'fashion_mnist', 'cifar10']:
            if os.path.exists(os.path.join(RESULTS_DIR, d)):
                datasets.append(d)
        if not datasets:
            print("No experiment results found. Run run_ps_experiment.py first.")
            return

    print(f"Generating figures for datasets: {datasets}, M={M}")
    fig1_disparity_histograms(datasets, M)
    fig2_pssd_vs_rii(datasets, M)
    fig3_roc_curves(datasets, M)
    fig4_topk_disparity(datasets, M)
    fig5_ps_mhpr_jensen(datasets, M)
    print(f"\nAll figures saved to {OUTPUT_DIR}/")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PSSD Visualization')
    parser.add_argument('--datasets', type=str, default='mnist,fashion_mnist,cifar10')
    parser.add_argument('--M', type=int, default=20)
    args = parser.parse_args()

    datasets = args.datasets.split(',')
    generate_all_figures(datasets, args.M)


if __name__ == '__main__':
    main()
