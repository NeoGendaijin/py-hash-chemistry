#!/usr/bin/env python3
"""
Runaway transition vs grid size L for one or more scans (per-seed CSVs in <root>/L{L}/runs).

Runaway = final mean component size > 100 (the paper's rule; runs stop at mean >= 2000).
Per L: runaway fraction with Wilson 95% interval, onset step (first sample with mean > 100),
fill fraction s_max/L^2 of runaway runs, final mean size / occupancy of non-runaway runs.

Example:
  python scripts/analyze_size_scan.py \
      --scan "corrected (CPU)=results/cpu_scan/original_mix" \
      --scan "published (JAX)=results/transition_scan" --out results/cpu_scan/analysis
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#eb6834", "#2a78d6", "#1baf7a", "#4a3aa7"]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def scan_table(root: Path):
    rows = []
    for d in sorted(root.glob("L*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        if not re.fullmatch(r"L\d+", d.name):
            continue
        L = int(d.name[1:])
        runs = []
        for f in (d / "runs").glob("seed_*.csv"):
            if f.name.endswith(".partial.csv"):
                continue
            x = pd.read_csv(f)
            last = x.iloc[-1]
            over = x[x["mean_size"] > 100]
            runs.append(dict(runaway=bool(last["mean_size"] > 100),
                             onset=float(over["step"].iloc[0]) if len(over) else np.nan,
                             fill=float(last["max_size"]) / L**2,
                             mean_size=float(last["mean_size"]),
                             density=float(last["n_active"]) / L**2 if "n_active" in x else np.nan,
                             final_step=int(last["step"])))
        if not runs:
            continue
        r = pd.DataFrame(runs)
        k, n = int(r.runaway.sum()), len(r)
        lo, hi = wilson(k, n)
        nr = r[~r.runaway]
        rows.append(dict(L=L, n=n, runaway=k, frac=k / n, lo=lo, hi=hi,
                         onset_median=r.onset.median(), fill_median=r[r.runaway].fill.median(),
                         nonrunaway_mean_size=nr.mean_size.median(), nonrunaway_density=nr.density.median()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="append", required=True, help="label=path (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    md = ["# Runaway transition vs L\n"]
    for i, spec in enumerate(args.scan):
        label, path = spec.split("=", 1)
        t = scan_table(Path(path))
        t.to_csv(out / f"table_{i}.csv", index=False)
        c = COLORS[i % len(COLORS)]
        axes[0].errorbar(t.L, t.frac, yerr=[t.frac - t.lo, t.hi - t.frac], color=c, marker="o", ms=5, lw=1.8, capsize=3, label=label)
        axes[1].plot(t.L, t.onset_median, color=c, marker="o", ms=5, lw=1.8, label=label)
        axes[2].plot(t.L, t.nonrunaway_mean_size, color=c, marker="o", ms=5, lw=1.8, label=label)
        md.append(f"\n## {label} ({path})\n")
        md.append("| L | runs | runaway | fraction [95% CI] | onset median | fill (runaway) | non-runaway mean size | non-runaway density |")
        md.append("|---|---|---|---|---|---|---|---|")
        for _, r in t.iterrows():
            md.append(f"| {r.L} | {r.n} | {r.runaway} | {r.frac:.2f} [{r.lo:.2f}, {r.hi:.2f}] | {r.onset_median:.0f} | "
                      f"{r.fill_median:.3f} | {r.nonrunaway_mean_size:.2f} | {r.nonrunaway_density:.3f} |")
    axes[0].set_ylabel("Runaway fraction by step 20,000"); axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Median onset step (runaway runs)")
    axes[2].set_ylabel("Final mean component size (non-runaway)")
    for ax in axes:
        ax.set_xlabel("Grid size L")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"transition.{ext}", dpi=150)
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
