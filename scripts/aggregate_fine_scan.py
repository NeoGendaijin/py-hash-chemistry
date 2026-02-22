#!/usr/bin/env python3
"""Aggregate fine_transition_scan raw seed CSVs into summary.csv per L
and a combined transition analysis CSV."""

import os
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "results", "fine_transition_scan")
L_VALUES = list(range(300, 322, 2))  # 300, 302, ..., 320


def aggregate_one_L(L):
    """Load all seed CSVs for a given L, compute mean ± std per step."""
    runs_dir = os.path.join(BASE, f"L{L}", "runs")
    seed_files = sorted(
        f for f in os.listdir(runs_dir) if f.startswith("seed_") and f.endswith(".csv")
    )
    if not seed_files:
        print(f"  WARNING: No seed files for L={L}")
        return None, []

    dfs = []
    for sf in seed_files:
        df = pd.read_csv(os.path.join(runs_dir, sf))
        dfs.append(df)

    print(f"  L={L}: {len(dfs)} runs, {len(dfs[0])} steps")

    # Align on steps (all should share the same step column)
    steps = dfs[0]["step"].values
    metrics = ["max_fitness", "mean_fitness", "max_size", "mean_size",
               "cum_cell_types", "cum_pattern_types"]

    summary_rows = []
    for i, step in enumerate(steps):
        row = {"step": int(step)}
        for m in metrics:
            vals = np.array([df[m].iloc[i] for df in dfs])
            row[f"{m}_mean"] = vals.mean()
            row[f"{m}_std"] = vals.std()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save summary.csv
    out_path = os.path.join(BASE, f"L{L}", "summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}")

    return summary_df, dfs


def compute_analysis(all_summaries, all_raw):
    """Compute per-L final-step statistics (like transition_analysis.csv)."""
    rows = []
    for L in L_VALUES:
        if L not in all_summaries:
            continue
        summary = all_summaries[L]
        raw_dfs = all_raw[L]
        last = summary.iloc[-1]

        # Early window: steps 2000-5000, mid: 5000-10000, late: 15000-20000
        early_mask = (summary["step"] >= 2000) & (summary["step"] <= 5000)
        mid_mask = (summary["step"] >= 5000) & (summary["step"] <= 10000)
        late_mask = (summary["step"] >= 15000) & (summary["step"] <= 20000)

        mean_size_early = summary.loc[early_mask, "mean_size_mean"].mean()
        mean_size_mid = summary.loc[mid_mask, "mean_size_mean"].mean()
        mean_size_late = summary.loc[late_mask, "mean_size_mean"].mean()
        mean_fitness_early = summary.loc[early_mask, "mean_fitness_mean"].mean()
        mean_fitness_late = summary.loc[late_mask, "mean_fitness_mean"].mean()

        late_over_early = mean_size_late / mean_size_early if mean_size_early > 0 else np.nan

        # Correlation between size and fitness over steps 2000-10000
        corr_mask = (summary["step"] >= 2000) & (summary["step"] <= 10000)
        corr_data = summary.loc[corr_mask]
        if len(corr_data) > 2:
            corr = np.corrcoef(corr_data["mean_size_mean"], corr_data["mean_fitness_mean"])[0, 1]
        else:
            corr = np.nan

        # Slope of mean_size over 2000-10000
        if len(corr_data) > 2:
            x = corr_data["step"].values.astype(float)
            y = corr_data["mean_size_mean"].values
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = np.nan

        # Threshold crossing times from individual runs
        thresholds = [10, 100, 1000]
        t_cross = {}
        for thr in thresholds:
            times = []
            for df in raw_dfs:
                crossed = df[df["mean_size"] >= thr]
                if len(crossed) > 0:
                    times.append(crossed["step"].iloc[0])
            t_cross[thr] = np.mean(times) if times else np.nan

        # Runaway fraction (mean_size > 100 at final step)
        runaway_count = sum(1 for df in raw_dfs if df["mean_size"].iloc[-1] > 100)
        runaway_frac = runaway_count / len(raw_dfs)

        rows.append({
            "L": L,
            "mean_size_end": last["mean_size_mean"],
            "max_size_end": last["max_size_mean"],
            "mean_size_end_norm": last["mean_size_mean"] / (L * L),
            "max_size_end_norm": last["max_size_mean"] / (L * L),
            "mean_size_early": mean_size_early,
            "mean_size_mid": mean_size_mid,
            "mean_size_late": mean_size_late,
            "late_over_early": late_over_early,
            "mean_fitness_early": mean_fitness_early,
            "mean_fitness_late": mean_fitness_late,
            "mean_fitness_late_over_early": mean_fitness_late / mean_fitness_early if mean_fitness_early > 0 else np.nan,
            "corr_size_fitness_2k_10k": corr,
            "slope_mean_size_2k_10k": slope,
            "t_mean_size_10": t_cross[10],
            "t_mean_size_100": t_cross[100],
            "t_mean_size_1000": t_cross[1000],
            "patterns_end": last["cum_pattern_types_mean"],
            "max_size_peak": last["max_size_mean"],
            "mean_fitness_end": last["mean_fitness_mean"],
            "max_fitness_end": last["max_fitness_mean"],
            "runaway_fraction": runaway_frac,
            "runaway_count": runaway_count,
            "n_runs": len(raw_dfs),
            "mean_size_end_std": last["mean_size_std"],
        })

    analysis_df = pd.DataFrame(rows)
    return analysis_df


def main():
    print("Aggregating fine_transition_scan data...")
    all_summaries = {}
    all_raw = {}
    for L in L_VALUES:
        summary, raw = aggregate_one_L(L)
        if summary is not None:
            all_summaries[L] = summary
            all_raw[L] = raw

    # Save combined analysis
    os.makedirs(os.path.join(BASE, "analysis"), exist_ok=True)
    analysis = compute_analysis(all_summaries, all_raw)
    out_path = os.path.join(BASE, "analysis", "fine_transition_analysis.csv")
    analysis.to_csv(out_path, index=False)
    print(f"\nSaved combined analysis: {out_path}")
    print(analysis[["L", "mean_size_end", "max_size_end", "runaway_fraction",
                     "corr_size_fitness_2k_10k", "mean_fitness_end"]].to_string(index=False))


if __name__ == "__main__":
    main()
