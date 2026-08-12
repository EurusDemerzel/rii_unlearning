#!/usr/bin/env python3
"""Visualize supplement_v3 results -> results/supplement_v3/figures/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join("results", "supplement_v3", "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------- Fig A: RII threshold calibration ----------------
a = pd.read_csv(os.path.join("results", "supplement_v3", "expA_threshold.csv"))
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(a["steps"], a["rho"], "o-", color="#1f77b4", label=r"$\rho$ (gradient ascent)")
ax1.set_xlabel("Gradient-ascent steps")
ax1.set_ylabel(r"RII $\rho$", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.axhline(0.2333, ls="--", color="gray", lw=1, label="NoUnlearn ρ (no forgetting)")
ax1.axhline(0.0915, ls=":", color="green", lw=1, label="Retrain oracle ρ")
ax1.axhline(0.01, ls="-.", color="red", lw=1, label="suggested safe threshold 0.01")
ax1.legend(loc="upper left", fontsize=8)
ax2 = ax1.twinx()
ax2.plot(a["steps"], a["mia_loss_auc"], "s--", color="#d62728", alpha=0.7,
         label="MIA-loss AUC")
ax2.set_ylabel("MIA-loss AUC", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax2.legend(loc="lower right", fontsize=8)
ax1.set_title("RII threshold calibration (CIFAR-10, class-level, forget=cat)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figA_threshold.png"), dpi=150)
plt.close(fig)

# ---------------- Fig B: average-LOCO vs projection MHPR ----------------
b = pd.read_csv(os.path.join("results", "supplement_v3", "expB_loco.csv"))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
K = b["K"]
ax.plot(K, b["loco_single_min"], "v--", color="gray", label="single-ref LOCO (min)")
ax.plot(K, b["loco_single_max"], "^--", color="gray", label="single-ref LOCO (max)")
ax.plot(K, b["loco_avg"], "o-", color="#ff7f0e", label="average-ref LOCO")
ax.plot(K, b["mhpr_proj"], "s-", color="#2ca02c", label="projection MHPR")
ax.set_xlabel("Number of held-out references K")
ax.set_ylabel("projection residual (relative)")
ax.set_xticks(K)
ax.legend(fontsize=8)
ax.set_title("Average-reference LOCO vs projection MHPR (MNIST, forget=5)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figB_loco_vs_mhpr.png"), dpi=150)
plt.close(fig)

# ---------------- Fig C: cross-class gradient sweep ----------------
c = pd.read_csv(os.path.join("results", "supplement_v3", "expC_cross_class.csv"))
names = {2: "class 2 (bird)", 3: "class 3 (cat)", 5: "class 5 (dog)"}
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for cls, grp in c.groupby("forget_class"):
    g = grp.sort_values("steps")
    ax.plot(g["steps"], g["rho"], "o-", label=names[cls])
ax.axhline(0.2333, ls="--", color="gray", lw=1, label="NoUnlearn ρ (cat protocol)")
ax.set_xlabel("Gradient-ascent steps")
ax.set_ylabel(r"RII $\rho$")
ax.legend(fontsize=8)
ax.set_title("Cross-class gradient ascent: RII rises for all forget classes (CIFAR-10)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figC_cross_class.png"), dpi=150)
plt.close(fig)

print("figures written to", OUT)
