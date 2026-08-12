#!/usr/bin/env python3
"""Plot χ² validation results for Theorem B."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("results/mhpr_review/chi2_validation.json") as f:
    data = json.load(f)

steps = [d['steps'] for d in data]
rho = [d['rho_H'] for d in data]
chi2 = [d['chi2'] for d in data]
bound = [d['bound'] for d in data]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
ax.plot(steps, rho, 'o-', color='#E74C3C', linewidth=2)
ax.set_xlabel('Gradient ascent steps', fontsize=12)
ax.set_ylabel('ρ_H (MHPR)', fontsize=12)
ax.set_title('State-Space MHPR', fontsize=13)
ax.grid(alpha=0.3)
ax.set_xscale('symlog')

ax = axes[1]
ax.plot(steps, chi2, 's-', color='#3498DB', linewidth=2, label='χ²')
ax.set_xlabel('Gradient ascent steps', fontsize=12)
ax.set_ylabel('χ² divergence', fontsize=12)
ax.set_title('χ² Divergence', fontsize=13)
ax.grid(alpha=0.3)
ax.set_xscale('symlog')

ax = axes[2]
ax.plot(steps, rho, 'o-', color='#E74C3C', linewidth=2, label='ρ_H')
ax.plot(steps, bound, 's--', color='#2ECC71', linewidth=2, label='2χ²/||π_f||²')
ax.fill_between(steps, rho, bound, alpha=0.15, color='#2ECC71')
ax.set_xlabel('Gradient ascent steps', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('ρ_H vs Upper Bound', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xscale('symlog')

plt.tight_layout()
os.makedirs("results/mhpr_review", exist_ok=True)
plt.savefig("results/mhpr_review/chi2_validation.png", dpi=150)
print("Saved to results/mhpr_review/chi2_validation.png")
plt.close()
