#!/usr/bin/env python3
"""
Parameter sensitivity analysis for SCHC.

Tests whether the transition persists under different parameter values.
Reuses run_single and aggregation from large_space_scaling.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from jax import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import SCHCParams, advance_one_step_jit, initialize_state
from scripts.large_space_scaling import run_single, _save_run_csv, _load_runs, METRIC_KEYS


def aggregate_runs(size_dir: Path, size: int) -> Dict[str, np.ndarray] | None:
    """Aggregate runs and return summary dict."""
    run_dir = size_dir / "runs"
    runs = _load_runs(run_dir)
    if not runs:
        return None

    steps = runs[0]["step"]
    summary: Dict[str, np.ndarray] = {"step": steps}

    for key in METRIC_KEYS:
        stacked = np.stack([r[key] for r in runs], axis=0)
        summary[f"{key}_mean"] = stacked.mean(axis=0)
        summary[f"{key}_std"] = stacked.std(axis=0)

    # Save summary CSV
    summary_cols = ["step"]
    for key in METRIC_KEYS:
        summary_cols.append(f"{key}_mean")
        summary_cols.append(f"{key}_std")
    summary_data = np.column_stack([summary[col] for col in summary_cols])
    fmt = ["%d"] + ["%.8f"] * (len(summary_cols) - 1)
    np.savetxt(size_dir / "summary.csv", summary_data, delimiter=",",
               header=",".join(summary_cols), comments="", fmt=fmt)

    return summary


def run_sensitivity(param_name: str, param_values: List[float],
                    sizes: List[int], runs: int, steps: int,
                    base_k: int, base_n: int, base_mu: float,
                    base_death: float, output_root: Path,
                    seed0: int = 0, progress_interval: int = 500):
    """Run sensitivity analysis for a single parameter."""
    output_root.mkdir(parents=True, exist_ok=True)

    all_summaries: Dict[float, Dict[int, Dict[str, np.ndarray]]] = {}

    for pval in param_values:
        pval_dir = output_root / f"{param_name}_{pval:.4f}"
        pval_dir.mkdir(parents=True, exist_ok=True)

        if param_name == "mu":
            mu_val = pval
            death_val = base_death
        elif param_name == "death_prob":
            mu_val = base_mu
            death_val = pval
        else:
            raise ValueError(f"Unknown param: {param_name}")

        size_summaries = {}
        for size in sizes:
            params = SCHCParams(k=base_k, n=base_n, L=size, mu=mu_val, death_prob=death_val)
            size_dir = pval_dir / f"L{size}"
            run_dir = size_dir / "runs"
            size_dir.mkdir(parents=True, exist_ok=True)

            for i in range(runs):
                seed = seed0 + i
                outpath = run_dir / f"seed_{seed}.csv"
                if outpath.exists():
                    print(f"[skip] {outpath} exists")
                    continue
                print(f"[run] {param_name}={pval:.4f} L={size} seed={seed}")
                metrics = run_single(params, steps, seed, progress_interval=progress_interval)
                _save_run_csv(metrics, run_dir, seed)

            summary = aggregate_runs(size_dir, size)
            if summary is not None:
                size_summaries[size] = summary

        all_summaries[pval] = size_summaries

    # Generate comparison plots
    _plot_sensitivity(all_summaries, param_name, param_values, sizes, output_root)


def _plot_sensitivity(all_summaries, param_name, param_values, sizes, output_root):
    """Generate plots comparing different parameter values."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(param_values)))

    # Plot 1: Final mean_size vs L for each parameter value
    fig, ax = plt.subplots(figsize=(8, 5))
    for pval, color in zip(param_values, colors):
        if pval not in all_summaries or not all_summaries[pval]:
            continue
        L_vals = []
        mean_sizes = []
        std_sizes = []
        for size in sizes:
            if size in all_summaries[pval]:
                summary = all_summaries[pval][size]
                L_vals.append(size)
                mean_sizes.append(summary["mean_size_mean"][-1])
                std_sizes.append(summary["mean_size_std"][-1])
        if L_vals:
            ax.errorbar(L_vals, mean_sizes, yerr=std_sizes, marker="o",
                        label=f"{param_name}={pval:.4f}", color=color, capsize=3)

    ax.set_xlabel("L (grid size)")
    ax.set_ylabel("Mean component size (final step)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Mean Size vs L (varying {param_name})")
    plt.tight_layout()
    plt.savefig(output_root / f"mean_size_vs_L_{param_name}.png", dpi=200)
    plt.close()

    # Plot 2: Final mean_fitness vs L
    fig, ax = plt.subplots(figsize=(8, 5))
    for pval, color in zip(param_values, colors):
        if pval not in all_summaries or not all_summaries[pval]:
            continue
        L_vals = []
        mean_fits = []
        std_fits = []
        for size in sizes:
            if size in all_summaries[pval]:
                summary = all_summaries[pval][size]
                L_vals.append(size)
                mean_fits.append(summary["mean_fitness_mean"][-1])
                std_fits.append(summary["mean_fitness_std"][-1])
        if L_vals:
            ax.errorbar(L_vals, mean_fits, yerr=std_fits, marker="o",
                        label=f"{param_name}={pval:.4f}", color=color, capsize=3)

    ax.set_xlabel("L (grid size)")
    ax.set_ylabel("Mean fitness (final step)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Mean Fitness vs L (varying {param_name})")
    plt.tight_layout()
    plt.savefig(output_root / f"mean_fitness_vs_L_{param_name}.png", dpi=200)
    plt.close()

    # Plot 3: Time series comparison for a reference size (L=320)
    ref_size = 320
    if any(ref_size in all_summaries.get(pval, {}) for pval in param_values):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for pval, color in zip(param_values, colors):
            if pval not in all_summaries or ref_size not in all_summaries[pval]:
                continue
            summary = all_summaries[pval][ref_size]
            steps = summary["step"]
            axes[0].plot(steps, summary["mean_size_mean"],
                         label=f"{param_name}={pval:.4f}", color=color, linewidth=1.5)
            axes[1].plot(steps, summary["mean_fitness_mean"],
                         label=f"{param_name}={pval:.4f}", color=color, linewidth=1.5)

        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Mean component size")
        axes[0].set_title(f"Mean Size at L={ref_size} (varying {param_name})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Mean fitness")
        axes[1].set_title(f"Mean Fitness at L={ref_size} (varying {param_name})")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_root / f"timeseries_L{ref_size}_{param_name}.png", dpi=200)
        plt.close()

    # Save summary table
    rows = []
    for pval in param_values:
        for size in sizes:
            if pval in all_summaries and size in all_summaries[pval]:
                summary = all_summaries[pval][size]
                rows.append({
                    param_name: pval,
                    "L": size,
                    "mean_size_final": summary["mean_size_mean"][-1],
                    "max_size_final": summary["max_size_mean"][-1],
                    "mean_fitness_final": summary["mean_fitness_mean"][-1],
                })
    if rows:
        cols = list(rows[0].keys())
        with open(output_root / f"sensitivity_summary_{param_name}.csv", "w") as f:
            f.write(",".join(cols) + "\n")
            for row in rows:
                f.write(",".join(str(row[c]) for c in cols) + "\n")

    print(f"[sensitivity] Saved all results to {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Parameter sensitivity analysis for SCHC.")
    parser.add_argument("--param", type=str, required=True, choices=["mu", "death_prob"],
                        help="Parameter to vary")
    parser.add_argument("--values", type=float, nargs="+", required=True,
                        help="Values to test")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 300, 320, 400])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--mu", type=str, default="0.002/0.999")
    parser.add_argument("--death-prob", type=float, default=0.001)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=500)
    args = parser.parse_args()

    base_mu = float(eval(str(args.mu), {"__builtins__": None}, {}))

    run_sensitivity(
        param_name=args.param,
        param_values=args.values,
        sizes=args.sizes,
        runs=args.runs,
        steps=args.steps,
        base_k=args.k,
        base_n=args.n,
        base_mu=base_mu,
        base_death=args.death_prob,
        output_root=args.output_root,
        seed0=args.seed0,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
