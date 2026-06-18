#!/usr/bin/env python3
"""Compare the size transition under open vs periodic (toroidal) boundaries.

Reads per-seed final mean component sizes from:
  open     : results/transition_scan/L{L}/runs/seed_*.csv      (existing scan)
  periodic : results/boundary_control/periodic/L{L}/runs/seed_*.csv

For each L it reports the runaway fraction (final mean size > 100) and the
median final mean size, and writes a comparison figure.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "tex" / "Large-Hash-Chemistry" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
DOUBLE_COL = 180 * MM
PALETTE = sns.color_palette("colorblind", 10)
RUNAWAY_THRESHOLD = 100.0

OPEN_ROOT = ROOT / "results" / "transition_scan"
PERIODIC_ROOT = ROOT / "results" / "boundary_control" / "periodic"


def final_mean_sizes(root: Path, L: int):
    runs = root / f"L{L}" / "runs"
    if not runs.is_dir():
        return None
    vals = []
    for f in sorted(runs.glob("seed_*.csv")):
        df = pd.read_csv(f)
        if len(df):
            vals.append(float(df["mean_size"].iloc[-1]))
    return np.array(vals, dtype=float) if vals else None


def summarize(root: Path, sizes):
    rows = {}
    for L in sizes:
        v = final_mean_sizes(root, L)
        if v is None or len(v) == 0:
            continue
        rows[L] = {
            "n": len(v),
            "runaway_frac": float(np.mean(v > RUNAWAY_THRESHOLD)),
            "median": float(np.median(v)),
            "vals": v,
        }
    return rows


def main():
    sizes = [200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]
    open_s = summarize(OPEN_ROOT, sizes)
    peri_s = summarize(PERIODIC_ROOT, sizes)

    print(f"{'L':>5} | {'open runaway':>12} {'open med':>10} | {'peri runaway':>12} {'peri med':>10}")
    for L in sizes:
        o = open_s.get(L); p = peri_s.get(L)
        os_ = f"{o['runaway_frac']:.2f}({o['n']})" if o else "-"
        om = f"{o['median']:.1f}" if o else "-"
        ps_ = f"{p['runaway_frac']:.2f}({p['n']})" if p else "-"
        pm = f"{p['median']:.1f}" if p else "-"
        print(f"{L:>5} | {os_:>12} {om:>10} | {ps_:>12} {pm:>10}")

    # ---- figure: runaway fraction and median final size vs L, open vs periodic ----
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.42), layout="constrained")

    def series(summ, key):
        Ls = sorted(summ.keys())
        return Ls, [summ[L][key] for L in Ls]

    # Panel (a): runaway fraction
    ax = axes[0]
    if open_s:
        Ls, ys = series(open_s, "runaway_frac")
        ax.plot(Ls, ys, "o-", color=PALETTE[0], label="Open")
    if peri_s:
        Ls, ys = series(peri_s, "runaway_frac")
        ax.plot(Ls, ys, "s--", color=PALETTE[3], label="Periodic")
    ax.set_xlabel("Space size $L$")
    ax.set_ylabel("Runaway fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, fontsize=8)
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    # Panel (b): median final mean size (log)
    ax = axes[1]
    if open_s:
        Ls, ys = series(open_s, "median")
        ax.plot(Ls, ys, "o-", color=PALETTE[0], label="Open")
    if peri_s:
        Ls, ys = series(peri_s, "median")
        ax.plot(Ls, ys, "s--", color=PALETTE[3], label="Periodic")
    ax.set_yscale("log")
    ax.set_xlabel("Space size $L$")
    ax.set_ylabel("Median final mean size")
    ax.legend(frameon=True, fontsize=8)
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    out = FIG_DIR / "fig_boundary_control"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png", dpi=300)
    print(f"\nSaved {out}.pdf / .png")


if __name__ == "__main__":
    main()
