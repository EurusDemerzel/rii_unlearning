"""
Visualization module for the machine unlearning pipeline.

Generates four plots:
  (a) Mutual Information vs Forget Ratio (by method)
  (b) σ₂/σ₁ Channel Rank vs Forget Ratio (by method)
  (c) MIA Attack Success Rate vs Forget Ratio
  (d) KL Divergence heatmap (method × forget ratio)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ── Colour palette ────────────────────────────────────────────────────────
METHOD_COLORS = {
    "NoUnlearning":  "#E74C3C",
    "Retrain":       "#27AE60",
    "SISA":          "#3498DB",
    "FineTune":      "#F39C12",
}
METHOD_MARKERS = {
    "NoUnlearning":  "s",
    "Retrain":       "o",
    "SISA":          "^",
    "FineTune":      "D",
}


def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)


def plot_mutual_information(df, output_path):
    """Line plot: Mutual Information vs Forget Ratio, coloured by method."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("forget_ratio")
        ax.plot(sub["forget_ratio"] * 100, sub["mutual_information"],
                color=METHOD_COLORS.get(method, "#999"),
                marker=METHOD_MARKERS.get(method, "o"),
                linewidth=2, markersize=8, label=method)
    _style_ax(ax, "Forget Ratio (%)", "Mutual Information I(X;Y) (nats)",
              "Mutual Information vs Forget Ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📈  MI plot saved → {output_path}")


def plot_sigma_ratio(df, output_path):
    """Line plot: σ₂/σ₁ vs Forget Ratio, coloured by method."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("forget_ratio")
        ax.plot(sub["forget_ratio"] * 100, sub["sigma_ratio"],
                color=METHOD_COLORS.get(method, "#999"),
                marker=METHOD_MARKERS.get(method, "o"),
                linewidth=2, markersize=8, label=method)
    _style_ax(ax, "Forget Ratio (%)", "Singular Value Ratio σ₂/σ₁",
              "Channel Rank Metric (σ₂/σ₁ → 0 = Rank-1)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📈  σ₂/σ₁ plot saved → {output_path}")


def plot_mia(df, output_path):
    """Line plot: MIA attack success rate vs Forget Ratio."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("forget_ratio")
        ax.plot(sub["forget_ratio"] * 100, sub["mia_best_acc"] * 100,
                color=METHOD_COLORS.get(method, "#999"),
                marker=METHOD_MARKERS.get(method, "o"),
                linewidth=2, markersize=8, label=method)
    # Random-guessing baseline
    ratios = sorted(df["forget_ratio"].unique()) * 100
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1.5,
               label="Random Guess (50%)")
    _style_ax(ax, "Forget Ratio (%)", "MIA Attack Accuracy (%)",
              "Membership Inference Attack Success Rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📈  MIA plot saved → {output_path}")


def plot_kl_heatmap(df, output_path):
    """Heatmap: KL divergence (method × forget_ratio)."""
    pivot = df.pivot_table(values="kl_symmetric", index="method",
                            columns="forget_ratio", aggfunc="mean")
    # Sort columns by forget_ratio
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(10, len(pivot) * 1.2 + 1))
    annot = pivot.map(lambda x: f"{x:.4f}")
    sns.heatmap(pivot, annot=annot, fmt="", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Symmetric KL Divergence"})
    ax.set_title("KL Divergence: Method × Forget Ratio", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("Forget Ratio", fontsize=12)
    ax.set_ylabel("Method", fontsize=12)
    # Format x-tick labels as percentages
    ax.set_xticklabels([f"{float(t.get_text())*100:.0f}%"
                         for t in ax.get_xticklabels()])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📈  KL heatmap saved → {output_path}")


def generate_all_plots(df, output_dir):
    """Generate all four plots and save them into `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)

    plot_mutual_information(df, os.path.join(output_dir, "mi_vs_forget_ratio.png"))
    plot_sigma_ratio(df,       os.path.join(output_dir, "sigma_ratio.png"))
    plot_mia(df,               os.path.join(output_dir, "mia_accuracy.png"))
    plot_kl_heatmap(df,        os.path.join(output_dir, "kl_heatmap.png"))
