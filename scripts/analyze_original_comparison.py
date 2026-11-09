#!/usr/bin/env python3
"""
Compare the JAX runs from scripts/compare_original_schc.py with the aggregated
data of the original Mathematica study (Sayama, SCHC, IEEE ALIFE-CIS 2025;
github.com/hsayama/Structural-Cellular-Hash-Chemistry, *Series.dat: 87 runs
that passed the >1 MB log-size filter, 2000 steps each).

The same filter is applied to the JAX runs (estimated Mathematica log size
> 1e6 bytes). Fitness curves follow the original plots: the ensemble mean of
the fitness is shown as -log10(1 - mean fitness).

Outputs (default results/original_comparison/analysis/):
  comparison.png / .pdf   seven panels, ensemble mean with 95% bootstrap bands
  summary.csv             per metric x arm x time window: mean, CI, ratio to original, Mann-Whitney p
  report.md               human-readable summary
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]

# (label, JAX column, original .dat file, kind)
METRICS = [
    ("Max fitness  -log10(1 - f)", "max_fitness_gap", "maxFitnessSeries.dat", "gap"),
    ("Mean fitness  -log10(1 - f)", "mean_fitness_gap", "meanFitnessSeries.dat", "gap"),
    ("Replicated active cells per step", "num_replicated_cells", "numIndivEntitiesSeries.dat", "level"),
    ("Max cells per replication event", "max_rep_size", "maxRepEventSeries.dat", "level"),
    ("Mean cells per replication event", "mean_rep_size", "meanRepEventSeries.dat", "level"),
    ("Cumulative cell types", "cum_cell_types", "nptypesSeries.dat", "level"),
    ("Cumulative pattern types", "cum_pattern_types", "nctypesSeries.dat", "level"),
]
ARMS = [  # (name, legend label, color: reference categorical slots 1-3)
    ("legacy", "JAX, published rule (fitter wins 3/4)", "#2a78d6"),
    ("fixed", "JAX, original winner rule", "#eb6834"),
    ("fixed_mix", "JAX, original rule + mixed score", "#1baf7a"),
]
REF_COLOR = "#0b0b0b"
WINDOWS = [(0, 10), (10, 100), (100, 1000), (1000, 2000), (1500, 2000)]
LOG_LIMIT = 1_000_000  # the original analysis keeps runs whose log file exceeds 1 MB


def load_dat(path: Path) -> np.ndarray:
    runs = []
    for line in path.read_text().strip().split("\n"):
        runs.append([float(v) for _, v in re.findall(r"\{([^,]+),\s*([^}]+)\}", line)])
    return np.array(runs)


def load_arm(arm_dir: Path):
    files = sorted(arm_dir.glob("seed_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    data = [np.genfromtxt(f, delimiter=",", names=True) for f in files]
    passed = [d for d in data if d["log_bytes_cum"][-1] > LOG_LIMIT]
    return {m[1]: np.array([d[m[1]] for d in passed]) for m in METRICS}, len(passed), len(data)


def ensemble_curve(x: np.ndarray, kind: str) -> np.ndarray:
    """Per-step ensemble mean, on the plotted scale."""
    m = np.nanmean(x, axis=0)
    return -np.log10(m) if kind == "gap" else m


def bootstrap_band(x: np.ndarray, kind: str, n_boot: int, rng) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.integers(0, x.shape[0], size=(n_boot, x.shape[0]))
    curves = np.array([ensemble_curve(x[i], kind) for i in idx])
    return np.nanpercentile(curves, 2.5, axis=0), np.nanpercentile(curves, 97.5, axis=0)


def window_values(x: np.ndarray, kind: str, lo: int, hi: int) -> np.ndarray:
    """Per-run window average on the plotted scale (for per-run tests)."""
    w = np.nanmean(x[:, lo:hi], axis=1)
    return -np.log10(w) if kind == "gap" else w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original-dir", default=str(ROOT.parent / "Structural-Cellular-Hash-Chemistry"))
    ap.add_argument("--runs-root", default=str(ROOT / "results/corner_fixed/original_comparison"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir) if args.out_dir else runs_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # Original: fitness series are stored as fitness, convert to the gap 1 - f.
    orig = {}
    for _, col, dat, kind in METRICS:
        x = load_dat(Path(args.original_dir) / dat)
        orig[col] = 1.0 - x if kind == "gap" else x
    n_orig = orig[METRICS[0][1]].shape[0]
    arms = {}
    for name, _, _ in ARMS:
        if (runs_root / name).is_dir():
            arms[name] = load_arm(runs_root / name)

    # Figure
    t = np.arange(1, 2001)  # original plots use time stamps 1..2000
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    for ax, (label, col, _, kind) in zip(axes.flat, METRICS):
        series = [("Mathematica original (n=%d)" % n_orig, orig[col], REF_COLOR, 2.4)]
        series += [(f"{lab} (n={arms[a][1]})", arms[a][0][col], c, 1.6) for a, lab, c in ARMS if a in arms]
        for lab, x, c, lw in series:
            lo, hi = bootstrap_band(x, kind, args.n_boot // 4, rng)
            ax.fill_between(t, lo, hi, color=c, alpha=0.15, linewidth=0)
            ax.plot(t, ensemble_curve(x, kind), color=c, lw=lw, label=lab)
        ax.set_xscale("log")
        ax.set_xlim(1, 2000)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Time step")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=10.5,
                         title="Ensemble mean, 95% bootstrap band", title_fontsize=10.5)
    fig.suptitle("SCHC at L=100, 2000 steps: JAX variants vs. the original Mathematica runs", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"comparison.{ext}", dpi=150)

    # Summary table: window means with bootstrap CI, ratio to original, Mann-Whitney on per-run window averages
    rows = ["metric,arm,window,mean,ci_lo,ci_hi,orig_mean,ratio_or_diff,mwu_p"]
    passing = "; ".join([f"Mathematica {n_orig}/100"] + [f"{a} {arms[a][1]}/{arms[a][2]}" for a, _, _ in ARMS if a in arms])
    md = ["# JAX SCHC vs. original Mathematica SCHC (L=100, 2000 steps)\n",
          f"Runs passing the >1 MB log filter: {passing}\n",
          "Values: ensemble mean over the window on the plotted scale "
          "(fitness: -log10(1 - mean f), higher = fitter). "
          "Bracket = ratio to original (difference for fitness); p = Mann-Whitney U on per-run window averages."]
    for label, col, _, kind in METRICS:
        md.append(f"\n## {label}\n\n| window | original | " + " | ".join(a for a, _, _ in ARMS if a in arms) + " |")
        md.append("|---|---|" + "---|" * len(arms))
        for lo, hi in WINDOWS:
            ov = float(np.nanmean(ensemble_curve(orig[col], kind)[lo:hi]))
            ow = window_values(orig[col], kind, lo, hi)
            cells = []
            for a, _, _ in ARMS:
                if a not in arms:
                    continue
                x = arms[a][0][col]
                v = float(np.nanmean(ensemble_curve(x, kind)[lo:hi]))
                idx = rng.integers(0, x.shape[0], size=(args.n_boot, x.shape[0]))
                boots = [np.nanmean(ensemble_curve(x[i], kind)[lo:hi]) for i in idx]
                ci = np.nanpercentile(boots, [2.5, 97.5])
                rel = v - ov if kind == "gap" else v / ov
                p = mannwhitneyu(window_values(x, kind, lo, hi), ow, nan_policy="omit").pvalue
                rows.append(f"{col},{a},{lo}-{hi},{v:.6g},{ci[0]:.6g},{ci[1]:.6g},{ov:.6g},{rel:.6g},{p:.3g}")
                rel_s = f"{rel:+.2f}" if kind == "gap" else f"x{rel:.2f}"
                cells.append(f"{v:.3g} [{rel_s}] p={p:.2g}")
            md.append(f"| {lo}-{hi} | {ov:.3g} | " + " | ".join(cells) + " |")
    (out_dir / "summary.csv").write_text("\n".join(rows) + "\n")
    (out_dir / "report.md").write_text("\n".join(md) + "\n")
    print((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
