#!/usr/bin/env python3
"""
Statistical analysis of transition scan data (L=200..400).

Computes:
- Mean +/- std across replicates for each L and metric
- Mann-Whitney U test and t-test comparing L=300 vs L=320
- Effect sizes (Cohen's d)
- Fraction of runs entering the "runaway regime" at the final sampled step (mean_size > 100)
- Distribution of waiting times to threshold crossing
- Generates plots with error bars and confidence bands
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_seed_csvs(run_dir: Path) -> List[Dict[str, np.ndarray]]:
    """Load all seed CSVs from a run directory."""
    runs = []
    for path in sorted(run_dir.glob("seed_*.csv")):
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr[None, :]
        runs.append({
            "step": arr[:, 0].astype(np.int32),
            "max_fitness": arr[:, 1],
            "mean_fitness": arr[:, 2],
            "max_size": arr[:, 3],
            "mean_size": arr[:, 4],
            "cum_cell_types": arr[:, 5],
            "cum_pattern_types": arr[:, 6],
        })
    return runs


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float("nan")
    return float((group1.mean() - group2.mean()) / pooled_std)


def first_crossing_time(steps: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """Find the first step where values >= threshold."""
    idx = np.where(values >= threshold)[0]
    return float(steps[idx[0]]) if idx.size else float("nan")


def run_analysis(input_root: Path, output_dir: Path,
                 compare_sizes: Tuple[int, int] = (300, 320),
                 runaway_threshold: float = 100.0,
                 crossing_thresholds: List[float] = None):
    if crossing_thresholds is None:
        crossing_thresholds = [10, 50, 100, 500, 1000]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover available sizes
    size_dirs = sorted(input_root.glob("L*"))
    sizes = []
    all_runs: Dict[int, List[Dict[str, np.ndarray]]] = {}

    for sd in size_dirs:
        try:
            size = int(sd.name[1:])
        except ValueError:
            continue
        run_dir = sd / "runs"
        if not run_dir.exists():
            continue
        runs = load_seed_csvs(run_dir)
        if runs:
            sizes.append(size)
            all_runs[size] = runs

    sizes.sort()
    print(f"Found sizes: {sizes}")
    print(f"Runs per size: {', '.join(f'L={s}: {len(all_runs[s])}' for s in sizes)}")

    # ---- Per-size summary statistics ----
    metrics_of_interest = ["mean_size", "max_size", "mean_fitness"]
    summary_rows = []

    for size in sizes:
        runs = all_runs[size]
        row = {"L": size, "n_runs": len(runs)}

        for metric in metrics_of_interest:
            # Final-step values across runs
            final_vals = np.array([r[metric][-1] for r in runs])
            row[f"{metric}_final_mean"] = final_vals.mean()
            row[f"{metric}_final_std"] = final_vals.std()
            row[f"{metric}_final_median"] = np.median(final_vals)

            # Late-window mean (last 25% of steps)
            late_vals = []
            for r in runs:
                n_steps = len(r[metric])
                late_start = int(n_steps * 0.75)
                late_vals.append(r[metric][late_start:].mean())
            late_arr = np.array(late_vals)
            row[f"{metric}_late_mean"] = late_arr.mean()
            row[f"{metric}_late_std"] = late_arr.std()

        # Runaway fraction based on the final sampled mean size.
        runaway_count = 0
        for r in runs:
            if r["mean_size"][-1] > runaway_threshold:
                runaway_count += 1
        row["runaway_fraction"] = runaway_count / len(runs)
        row["runaway_count"] = runaway_count

        # Crossing times
        for thr in crossing_thresholds:
            times = []
            for r in runs:
                t = first_crossing_time(r["step"], r["mean_size"], thr)
                times.append(t)
            times_arr = np.array(times)
            valid = times_arr[~np.isnan(times_arr)]
            row[f"crossing_{int(thr)}_mean"] = valid.mean() if len(valid) > 0 else float("nan")
            row[f"crossing_{int(thr)}_std"] = valid.std() if len(valid) > 0 else float("nan")
            row[f"crossing_{int(thr)}_fraction"] = len(valid) / len(times)

        summary_rows.append(row)

    # Save summary CSV
    cols = list(summary_rows[0].keys())
    with open(output_dir / "per_size_summary.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for row in summary_rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")

    # ---- Statistical tests comparing L=300 vs L=320 ----
    s1, s2 = compare_sizes
    report_lines = []
    report_lines.append(f"# Statistical Analysis Report")
    report_lines.append(f"")
    report_lines.append(f"## Data Overview")
    report_lines.append(f"- Sizes analyzed: {sizes}")
    report_lines.append(f"- Runs per size: {', '.join(f'L={s}: {len(all_runs[s])}' for s in sizes)}")
    report_lines.append(f"")

    if s1 in all_runs and s2 in all_runs:
        runs_a = all_runs[s1]
        runs_b = all_runs[s2]

        report_lines.append(f"## Statistical Tests: L={s1} vs L={s2}")
        report_lines.append(f"")

        for metric in metrics_of_interest:
            # Final-step comparison
            vals_a = np.array([r[metric][-1] for r in runs_a])
            vals_b = np.array([r[metric][-1] for r in runs_b])

            # Late-window comparison
            late_a = np.array([r[metric][int(len(r[metric]) * 0.75):].mean() for r in runs_a])
            late_b = np.array([r[metric][int(len(r[metric]) * 0.75):].mean() for r in runs_b])

            report_lines.append(f"### {metric}")
            report_lines.append(f"")

            # Final values
            report_lines.append(f"**Final step values:**")
            report_lines.append(f"- L={s1}: mean={vals_a.mean():.4f}, std={vals_a.std():.4f}, median={np.median(vals_a):.4f}")
            report_lines.append(f"- L={s2}: mean={vals_b.mean():.4f}, std={vals_b.std():.4f}, median={np.median(vals_b):.4f}")

            # t-test
            t_stat, t_p = stats.ttest_ind(vals_a, vals_b, equal_var=False)
            report_lines.append(f"- Welch's t-test: t={t_stat:.4f}, p={t_p:.2e}")

            # Mann-Whitney U
            u_stat, u_p = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
            report_lines.append(f"- Mann-Whitney U: U={u_stat:.1f}, p={u_p:.2e}")

            # Cohen's d
            d = cohens_d(vals_a, vals_b)
            report_lines.append(f"- Cohen's d: {d:.4f}")
            report_lines.append(f"")

            # Late-window values
            report_lines.append(f"**Late-window values (last 25%):**")
            report_lines.append(f"- L={s1}: mean={late_a.mean():.4f}, std={late_a.std():.4f}")
            report_lines.append(f"- L={s2}: mean={late_b.mean():.4f}, std={late_b.std():.4f}")
            t_stat2, t_p2 = stats.ttest_ind(late_a, late_b, equal_var=False)
            u_stat2, u_p2 = stats.mannwhitneyu(late_a, late_b, alternative='two-sided')
            d2 = cohens_d(late_a, late_b)
            report_lines.append(f"- Welch's t-test: t={t_stat2:.4f}, p={t_p2:.2e}")
            report_lines.append(f"- Mann-Whitney U: U={u_stat2:.1f}, p={u_p2:.2e}")
            report_lines.append(f"- Cohen's d: {d2:.4f}")
            report_lines.append(f"")

    # ---- Runaway regime analysis ----
    report_lines.append(f"## Runaway Regime Analysis (mean_size > {runaway_threshold})")
    report_lines.append(f"")
    for row in summary_rows:
        report_lines.append(f"- L={row['L']:.0f}: {row['runaway_count']:.0f}/{row['n_runs']:.0f} runs ({row['runaway_fraction']*100:.1f}%)")
    report_lines.append(f"")

    # ---- Crossing times ----
    report_lines.append(f"## Threshold Crossing Times")
    report_lines.append(f"")
    for thr in crossing_thresholds:
        report_lines.append(f"### mean_size >= {int(thr)}")
        for row in summary_rows:
            mean_t = row[f"crossing_{int(thr)}_mean"]
            std_t = row[f"crossing_{int(thr)}_std"]
            frac = row[f"crossing_{int(thr)}_fraction"]
            if np.isnan(mean_t):
                report_lines.append(f"- L={row['L']:.0f}: never crossed ({frac*100:.0f}% of runs)")
            else:
                report_lines.append(f"- L={row['L']:.0f}: mean={mean_t:.1f} +/- {std_t:.1f} steps ({frac*100:.0f}% of runs)")
        report_lines.append(f"")

    # Save report
    (output_dir / "statistical_report.md").write_text("\n".join(report_lines))

    # ---- Plots ----
    L_arr = np.array(sizes, dtype=float)

    # 1. Mean size with error bars
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label, color in [
        ("mean_size", "Mean size", "blue"),
        ("max_size", "Max size", "red"),
    ]:
        means = np.array([next(r for r in summary_rows if r["L"] == s)[f"{metric}_final_mean"] for s in sizes])
        stds = np.array([next(r for r in summary_rows if r["L"] == s)[f"{metric}_final_std"] for s in sizes])
        ax.errorbar(L_arr, means, yerr=stds, marker="o", label=label, color=color, capsize=4)
    ax.set_xlabel("L (grid size)")
    ax.set_ylabel("Component size at final step")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Component Size vs Grid Size (mean ± std)")
    plt.tight_layout()
    plt.savefig(output_dir / "size_vs_L_errorbars.png", dpi=200)
    plt.close()

    # 2. Mean hash score with error bars
    fig, ax = plt.subplots(figsize=(8, 5))
    means = np.array([next(r for r in summary_rows if r["L"] == s)["mean_fitness_final_mean"] for s in sizes])
    stds = np.array([next(r for r in summary_rows if r["L"] == s)["mean_fitness_final_std"] for s in sizes])
    ax.errorbar(L_arr, means, yerr=stds, marker="o", color="green", capsize=4)
    ax.set_xlabel("L (grid size)")
    ax.set_ylabel("Mean hash score at final step")
    ax.grid(True, alpha=0.3)
    ax.set_title("Mean Hash Score vs Grid Size (mean ± std)")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_score_vs_L_errorbars.png", dpi=200)
    plt.close()

    # 3. Runaway fraction
    fig, ax = plt.subplots(figsize=(8, 5))
    fracs = np.array([next(r for r in summary_rows if r["L"] == s)["runaway_fraction"] for s in sizes])
    ax.bar(L_arr, fracs, width=8, color="coral", edgecolor="black")
    ax.set_xlabel("L (grid size)")
    ax.set_ylabel(f"Fraction of runs with final mean_size > {runaway_threshold}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Runaway Regime Probability vs Grid Size")
    plt.tight_layout()
    plt.savefig(output_dir / "runaway_fraction_vs_L.png", dpi=200)
    plt.close()

    # 4. Time series with confidence bands for key sizes
    key_sizes = [s for s in [200, 300, 320, 400] if s in all_runs]
    if key_sizes:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for size in key_sizes:
            runs = all_runs[size]
            # Ensure equal length
            min_len = min(len(r["mean_size"]) for r in runs)
            stacked = np.stack([r["mean_size"][:min_len] for r in runs])
            mean_ts = stacked.mean(axis=0)
            std_ts = stacked.std(axis=0)
            steps = runs[0]["step"][:min_len]

            axes[0].plot(steps, mean_ts, label=f"L={size}", linewidth=1.5)
            axes[0].fill_between(steps, mean_ts - std_ts, mean_ts + std_ts, alpha=0.15)

        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Mean component size")
        axes[0].set_yscale("log")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Mean Size Time Series (± 1 std)")

        for size in key_sizes:
            runs = all_runs[size]
            min_len = min(len(r["mean_fitness"]) for r in runs)
            stacked = np.stack([r["mean_fitness"][:min_len] for r in runs])
            mean_ts = stacked.mean(axis=0)
            std_ts = stacked.std(axis=0)
            steps = runs[0]["step"][:min_len]

            axes[1].plot(steps, mean_ts, label=f"L={size}", linewidth=1.5)
            axes[1].fill_between(steps, mean_ts - std_ts, mean_ts + std_ts, alpha=0.15)

        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Mean hash score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title("Mean Hash Score Time Series (± 1 std)")

        plt.tight_layout()
        plt.savefig(output_dir / "timeseries_confidence_bands.png", dpi=200)
        plt.close()

    # 5. Crossing time distributions (box plots)
    for thr in [10, 100]:
        fig, ax = plt.subplots(figsize=(10, 5))
        crossing_data = []
        crossing_labels = []
        for size in sizes:
            runs = all_runs[size]
            times = []
            for r in runs:
                t = first_crossing_time(r["step"], r["mean_size"], thr)
                if not np.isnan(t):
                    times.append(t)
            if times:
                crossing_data.append(times)
                crossing_labels.append(f"L={size}")

        if crossing_data:
            ax.boxplot(crossing_data, tick_labels=crossing_labels)
            ax.set_xlabel("Grid size")
            ax.set_ylabel(f"Steps to mean_size >= {int(thr)}")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_title(f"Distribution of Waiting Times (threshold={int(thr)})")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f"crossing_time_boxplot_thr{int(thr)}.png", dpi=200)
            plt.close()

    print(f"[analysis] All output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of transition scan data.")
    parser.add_argument("--input-root", type=Path, default=ROOT / "results" / "transition_scan")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "statistical_analysis")
    parser.add_argument("--compare", type=int, nargs=2, default=[300, 320],
                        help="Two sizes to compare statistically")
    parser.add_argument("--runaway-threshold", type=float, default=100.0)
    args = parser.parse_args()

    # Also include fine_transition_scan if available
    fine_scan_root = ROOT / "results" / "fine_transition_scan"

    run_analysis(args.input_root, args.output_dir,
                 compare_sizes=tuple(args.compare),
                 runaway_threshold=args.runaway_threshold)

    # If fine transition scan data is available, analyze that too
    if fine_scan_root.exists() and any(fine_scan_root.glob("L*/runs/seed_*.csv")):
        fine_output = ROOT / "results" / "statistical_analysis" / "fine_scan"
        print(f"\n[analysis] Also analyzing fine transition scan data...")
        run_analysis(fine_scan_root, fine_output,
                     compare_sizes=(308, 312),
                     runaway_threshold=args.runaway_threshold)


if __name__ == "__main__":
    main()
