#!/usr/bin/env python3
"""
Summarize the (D, P) sweep from scripts/run_spatial_dyadic_sweep.py with the three
outcome measures of Sayama's ALIFE 2026 LBA (Fig. 1), and compare the two JAX
dyadic rules ("intended", "as_coded") with the Mathematica values.

Measures, as in spatial-dyadic-SCHC-systematic-sim-data-analysis(-plotting).nb:
  successful runs      log size within (1 MB, 300 MB) and no extinction before t=2000;
                       runs with a runaway giant component are excluded from all three
                       measures (as in the LBA / npj Sec. 3 analysis) and counted separately
  mean_rep_size_500    ensemble mean (over successful runs) of the per-step mean number
                       of active cells per replication event, averaged over the last 500 steps
  pattern_types        ensemble mean of the cumulative number of distinct winner patterns at t=2000

The P=0 conditions do not depend on the dyadic rule and are shared.
Uncertainty: 95% bootstrap intervals over runs (per condition, and for the mean over D).
Outputs (default results/spatial_dyadic/analysis/): summary.csv, by_P.png/.pdf, heatmaps.png/.pdf, report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D_VALUES = list(range(10, 101, 10))
# A run counts as runaway if a single replication event ever copies >= 10% of the grid
# (the giant component of a runaway fills ~0.65 of it; compact replicators copy < 20 cells).
RUNAWAY_SIZE = 1000
P_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MEASURES = [
    ("successful_runs", "Successful runs (of 30)\n(runaway runs excluded)"),
    ("mean_rep_size_final500", "Mean active cells per replication\nevent, final 500 steps"),
    ("cum_pattern_types_final", "Cumulative pattern types\nin successful replication (t=2000)"),
]
SOURCES = [  # (key, label, color, marker)
    ("mathematica", "Mathematica (LBA Fig. 1; swap typo)", "#0b0b0b", "o"),
    ("as_coded", "JAX, dyadic rule as coded (swap never runs)", "#2a78d6", "s"),
    ("intended", "JAX, dyadic rule as intended", "#eb6834", "^"),
]


def load_condition(cond_dir: Path):
    """Per-run data of one condition: success flags, final-500 rep-size curves, final pattern counts."""
    files = sorted(cond_dir.glob("seed_*.csv"))
    if not files:
        return None
    runs = []
    for f in files:
        d = np.genfromtxt(f, delimiter=",", names=True)
        ok = bool(1e6 < d["log_bytes_cum"][-1] < 3e8 and d["n_active"][-1] > 0 and len(d) == 2000)
        runaway = bool(np.nanmax(d["max_rep_size"]) >= RUNAWAY_SIZE)
        runs.append((ok, d["mean_rep_size"][-500:], float(d["cum_pattern_types"][-1]), runaway))
    return runs


def measures(runs, exclude_runaway: bool = True) -> dict:
    """The three LBA measures. Runaway runs are excluded by default, as in the LBA/npj Sec. 3
    analysis (explosive-nucleation runs were excluded there); their count is reported separately."""
    ok = [r for r in runs if r[0] and not (exclude_runaway and r[3])]
    out = {"successful_runs": float(len(ok)), "runaway_runs": float(sum(r[3] for r in runs))}
    if ok:
        out["mean_rep_size_final500"] = float(np.nanmean(np.nanmean(np.array([r[1] for r in ok]), axis=0)))
        out["cum_pattern_types_final"] = float(np.mean([r[2] for r in ok]))
    else:
        out["mean_rep_size_final500"] = out["cum_pattern_types_final"] = np.nan
    return out


def bootstrap(runs, rng, n_boot):
    """n_boot resamples (with replacement over runs) of the three measures."""
    idx = rng.integers(0, len(runs), size=(n_boot, len(runs)))
    return [measures([runs[i] for i in row]) for row in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results" / "spatial_dyadic"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    table, boots = {}, {}  # (source, D, P) -> measures / list of bootstrap measures
    ref = np.genfromtxt(root / "sayama_lba_fig1.csv", delimiter=",", names=True)
    for r in ref:
        table[("mathematica", int(r["D"]), round(float(r["P"]), 1))] = {m: float(r[m]) for m, _ in MEASURES}
    for rule in ("intended", "as_coded"):
        for D in D_VALUES:
            for P in P_VALUES:
                src_rule = "intended" if P == 0.0 else rule  # P=0 is shared
                runs = load_condition(root / src_rule / f"D{D}_P{P:.1f}")
                if runs is not None and len(runs) >= 30:
                    table[(rule, D, P)] = measures(runs)
                    boots[(rule, D, P)] = bootstrap(runs, rng, args.n_boot)

    def ci(key, m):
        v = [b[m] for b in boots[key]] if key in boots else []
        return np.nanpercentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)

    def pooled(src, P, m):
        """Mean over available D, with a bootstrap CI (JAX sources) from independent per-condition resamples."""
        keys = [(src, D, P) for D in D_VALUES if (src, D, P) in table]
        if not keys:
            return None
        mean = float(np.nanmean([table[k][m] for k in keys]))
        if src == "mathematica":
            return mean, np.nan, np.nan, len(keys)
        reps = np.nanmean(np.array([[b[m] for b in boots[k]] for k in keys]), axis=0)
        lo, hi = np.nanpercentile(reps, [2.5, 97.5])
        return mean, lo, hi, len(keys)

    lines = ["source,D,P," + ",".join(f"{m},{m}_ci_lo,{m}_ci_hi" for m, _ in MEASURES)]
    for key, m in sorted(table.items()):
        src, D, P = key
        lines.append(f"{src},{D},{P}," + ",".join(
            f"{m[k]:.6g},{ci(key, k)[0]:.6g},{ci(key, k)[1]:.6g}" for k, _ in MEASURES))
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n")

    # Figure 1: each measure vs P (mean over the D values available), per source
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (m, label) in zip(axes, MEASURES):
        for src, slabel, color, marker in SOURCES:
            pts = [(P, pooled(src, P, m)) for P in P_VALUES]
            pts = [(P, v) for P, v in pts if v is not None]
            if not pts:
                continue
            xs = [P for P, _ in pts]
            ys = [v[0] for _, v in pts]
            ax.plot(xs, ys, color=color, marker=marker, lw=2, ms=7, label=slabel)
            if src != "mathematica":
                ax.fill_between(xs, [v[1] for _, v in pts], [v[2] for _, v in pts], color=color, alpha=0.18, linewidth=0)
        ax.set_xlabel("Dyadic interaction probability P")
        ax.set_title(label, fontsize=11)
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Spatial-dyadic SCHC, L=100: mean over D (JAX band = 95% bootstrap CI over runs)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"by_P.{ext}", dpi=150)

    # Figure 2: heatmaps (P x D) per measure and source, shared color scale per measure
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for i, (m, label) in enumerate(MEASURES):
        vals = [v[m] for v in table.values() if m in v and np.isfinite(v[m])]
        vmin, vmax = (min(vals), max(vals)) if vals else (0, 1)
        for j, (src, slabel, _, _) in enumerate(SOURCES):
            grid = np.full((len(P_VALUES), len(D_VALUES)), np.nan)
            for a, P in enumerate(P_VALUES):
                for b, D in enumerate(D_VALUES):
                    if (src, D, P) in table:
                        grid[a, b] = table[(src, D, P)][m]
            ax = axes[i, j]
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(D_VALUES)), D_VALUES)
            ax.set_yticks(range(len(P_VALUES)), P_VALUES)
            ax.set_xlabel("D"); ax.set_ylabel("P")
            ax.set_title(f"{slabel}\n{label.splitlines()[0]}", fontsize=9.5)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"heatmaps.{ext}", dpi=130)

    # Report
    md = ["# Spatial-dyadic SCHC (L=100): Mathematica vs JAX dyadic rules\n",
          "Mean over D of each measure; JAX: [95% bootstrap CI over runs]; n = number of D values available.",
          "Mathematica values are single 30-run estimates (no per-run data), with sampling error comparable to the JAX CIs.\n"]
    md.append("\n## Runaway runs (a replication event >= %d cells), per condition\n" % RUNAWAY_SIZE)
    md.append("| rule | P | " + " | ".join(f"D={D}" for D in D_VALUES) + " |")
    md.append("|---|---|" + "---|" * len(D_VALUES))
    for rule in ("intended", "as_coded"):
        for P in P_VALUES:
            md.append(f"| {rule} | {P} | " + " | ".join(
                str(int(table[(rule, D, P)]["runaway_runs"])) if (rule, D, P) in table else "-" for D in D_VALUES) + " |")
    for m, label in MEASURES:
        md.append(f"\n## {label.replace(chr(10), ' ')}\n\n| P | " + " | ".join(s[1] for s in SOURCES) + " |")
        md.append("|---|" + "---|" * len(SOURCES))
        for P in P_VALUES:
            cells = []
            for src, *_ in SOURCES:
                v = pooled(src, P, m)
                if v is None:
                    cells.append("-")
                elif src == "mathematica":
                    cells.append(f"{v[0]:.3g} (n={v[3]})")
                else:
                    cells.append(f"{v[0]:.3g} [{v[1]:.3g}, {v[2]:.3g}] (n={v[3]})")
            md.append(f"| {P} | " + " | ".join(cells) + " |")
    (out_dir / "report.md").write_text("\n".join(md) + "\n")
    print((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
