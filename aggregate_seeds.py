#!/usr/bin/env python3
"""aggregate_seeds.py — average the per-seed benchmark CSVs into mean +/- std.

Usage:
  python aggregate_seeds.py results/benchmark_v2_cifar10_s0 results/benchmark_v2_cifar10_s1 ...
  python aggregate_seeds.py --dirs "results/benchmark_v2_cifar10_s*"

For each method and metric column, prints mean and std across the given seeds,
and writes <outdir>/aggregated.csv (mean + std columns).
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

METRIC_COLS = ["retain_acc", "forget_acc", "rii_rho", "mhpr",
               "mia_loss_auc", "repr_mmd", "residual_probe_auc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results/aggregated.csv")
    args = ap.parse_args()

    dirs = []
    for d in args.dirs:
        dirs.extend(sorted(glob.glob(d)))
    if not dirs:
        print("no result dirs matched"); sys.exit(1)

    frames = []
    for d in dirs:
        csv_path = os.path.join(d, "results.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] {d} (no results.csv)"); continue
        df = pd.read_csv(csv_path)
        df["seed_dir"] = os.path.basename(d)
        frames.append(df)
    if not frames:
        print("no frames loaded"); sys.exit(1)
    all_df = pd.concat(frames, ignore_index=True)

    rows = []
    for method, g in all_df.groupby("method"):
        r = {"method": method, "n_seeds": len(g)}
        for c in METRIC_COLS:
            if c in g.columns and g[c].notna().any():
                r[f"{c}_mean"] = g[c].mean()
                r[f"{c}_std"] = g[c].std() if len(g) > 1 else np.nan
        rows.append(r)
    agg = pd.DataFrame(rows).sort_values("method")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    agg.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    print(agg[["method", "n_seeds", "retain_acc_mean", "forget_acc_mean",
               "rii_rho_mean", "rii_rho_std", "mhpr_mean", "mia_loss_auc_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
