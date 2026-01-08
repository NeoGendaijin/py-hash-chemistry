"""
Run SCHC at multiple spatial sizes (100/200/400/800), save per-run numeric metrics,
and reproduce Fig.4–6 style plots for size-scaling analysis.

The script writes per-seed CSVs under results/large_space/L{size}/runs and can
optionally aggregate everything present in that directory into summary CSVs and plots.
Use --skip-aggregate when launching split seed batches across GPUs, then rerun with
--aggregate-only afterward to assemble the combined results.
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

METRIC_KEYS = ["max_fitness", "mean_fitness", "max_size", "mean_size", "cum_cell_types", "cum_pattern_types"]


def _fitness_numpy(component: List[Tuple[int, int, int]]) -> float:
    if not component:
        return 0.0
    arr = np.array(component, dtype=np.uint32)  # columns: x, y, val
    min_xy = arr[:, :2].min(axis=0)
    rel = arr[:, :2] - min_xy
    order = np.lexsort((rel[:, 1], rel[:, 0]))
    rel = rel[order]
    values = arr[:, 2][order]
    elements = (rel[:, 0] * np.uint32(0x9E3779B9)) ^ (rel[:, 1] * np.uint32(0x85EBCA6B)) ^ values
    mixed = np.bitwise_xor.reduce(elements, dtype=np.uint32).astype(np.uint64)
    mixed = (mixed ^ np.uint64(0x45D9F3B)) * np.uint64(0x27D4EB2D)
    mixed = mixed & np.uint64(0xFFFFFFFF)
    return float(mixed.astype(np.float32) / np.float32(np.iinfo(np.uint32).max))


def _component_signature(component: List[Tuple[int, int, int]]) -> Tuple[Tuple[int, int, int], ...]:
    """Canonical signature of component relative positions and values."""

    arr = np.array(component, dtype=np.int32)
    min_xy = arr[:, :2].min(axis=0)
    rel = arr[:, :2] - min_xy
    vals = arr[:, 2]
    order = np.lexsort((rel[:, 1], rel[:, 0]))
    rel = rel[order]
    vals = vals[order]
    return tuple((int(x), int(y), int(v)) for (x, y), v in zip(rel, vals))


def _components_and_metrics(config_np: np.ndarray) -> Dict[str, float]:
    """Connected components and summary metrics."""

    L = config_np.shape[0]
    visited = np.zeros_like(config_np, dtype=bool)
    sizes: List[int] = []
    fitnesses: List[float] = []
    num_active = int((config_np > 0).sum())
    pattern_signatures: List[Tuple[int, int, int]] = []
    cell_types: List[int] = []

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for x in range(L):
        for y in range(L):
            if config_np[x, y] == 0 or visited[x, y]:
                continue
            stack = [(x, y)]
            visited[x, y] = True
            comp: List[Tuple[int, int, int]] = []
            while stack:
                cx, cy = stack.pop()
                val = int(config_np[cx, cy])
                comp.append((cx, cy, val))
                cell_types.append(val)
                for dx, dy in neighbors:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < L and 0 <= ny < L and config_np[nx, ny] != 0 and not visited[nx, ny]:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            sizes.append(len(comp))
            fitnesses.append(_fitness_numpy(comp))
            pattern_signatures.append(_component_signature(comp))

    if not sizes:
        return {
            "active_cells": 0,
            "num_components": 0,
            "mean_size": 0.0,
            "max_size": 0.0,
            "mean_fitness": 0.0,
            "max_fitness": 0.0,
            "cell_types": set(),
            "pattern_types": set(),
        }

    fitness_arr = np.array(fitnesses, dtype=np.float32)
    size_arr = np.array(sizes, dtype=np.float32)
    return {
        "active_cells": num_active,
        "num_components": len(sizes),
        "mean_size": float(size_arr.mean()),
        "max_size": float(size_arr.max()),
        "mean_fitness": float(fitness_arr.mean()),
        "max_fitness": float(fitness_arr.max()),
        "cell_types": set(cell_types),
        "pattern_types": set(pattern_signatures),
    }


def run_single(params: SCHCParams, steps: int, seed: int, progress_interval: int = 0):
    key = random.PRNGKey(seed)
    state = initialize_state(key, params)

    metrics = {
        "max_fitness": [],
        "mean_fitness": [],
        "max_size": [],
        "mean_size": [],
        "cum_cell_types": [],
        "cum_pattern_types": [],
    }
    seen_cell_types: set[int] = set()
    seen_pattern_types: set[Tuple[int, int, int]] = set()

    for t in range(steps + 1):
        cfg_np = np.array(state.config)
        m = _components_and_metrics(cfg_np)
        metrics["max_fitness"].append(m["max_fitness"])
        metrics["mean_fitness"].append(m["mean_fitness"])
        metrics["max_size"].append(m["max_size"])
        metrics["mean_size"].append(m["mean_size"])
        seen_cell_types |= set(m["cell_types"])
        seen_pattern_types |= set(m["pattern_types"])
        metrics["cum_cell_types"].append(len(seen_cell_types))
        metrics["cum_pattern_types"].append(len(seen_pattern_types))

        if progress_interval and t % progress_interval == 0:
            print(f"[run seed={seed}] step={t}")
            sys.stdout.flush()

        if t == steps:
            break
        state = advance_one_step_jit(state, params)

    return metrics


def _save_run_csv(metrics: Dict[str, List[float]], run_dir: Path, seed: int) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = np.arange(len(metrics["max_fitness"]), dtype=np.int32)
    data = np.column_stack(
        [
            steps,
            np.array(metrics["max_fitness"], dtype=np.float64),
            np.array(metrics["mean_fitness"], dtype=np.float64),
            np.array(metrics["max_size"], dtype=np.float64),
            np.array(metrics["mean_size"], dtype=np.float64),
            np.array(metrics["cum_cell_types"], dtype=np.int64),
            np.array(metrics["cum_pattern_types"], dtype=np.int64),
        ]
    )
    outpath = run_dir / f"seed_{seed}.csv"
    header = "step,max_fitness,mean_fitness,max_size,mean_size,cum_cell_types,cum_pattern_types"
    fmt = ["%d", "%.8f", "%.8f", "%.8f", "%.8f", "%d", "%d"]
    np.savetxt(outpath, data, delimiter=",", header=header, comments="", fmt=fmt)
    return outpath


def _load_runs(run_dir: Path):
    run_files = sorted(run_dir.glob("seed_*.csv"))
    runs = []
    for path in run_files:
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr[None, :]
        runs.append(
            {
                "step": arr[:, 0].astype(np.int32),
                "max_fitness": arr[:, 1],
                "mean_fitness": arr[:, 2],
                "max_size": arr[:, 3],
                "mean_size": arr[:, 4],
                "cum_cell_types": arr[:, 5],
                "cum_pattern_types": arr[:, 6],
            }
        )
    return runs


def _plot_runs(runs: List[np.ndarray], outpath: Path, ylabel: str, log_time: bool = False):
    steps = np.arange(runs[0].shape[0])
    plt.figure(figsize=(6, 3))
    for r in runs:
        plt.plot(steps, r, color="red", alpha=0.25, linewidth=0.8)
    avg = np.mean(np.stack(runs, axis=0), axis=0)
    plt.plot(steps, avg, color="black", linewidth=1.5)
    if log_time:
        plt.xscale("log")
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _aggregate_and_plot(size_dir: Path, size: int) -> None:
    run_dir = size_dir / "runs"
    runs = _load_runs(run_dir)
    if not runs:
        print(f"[aggregate] No run CSVs found in {run_dir}; skipping.")
        return None

    steps = runs[0]["step"]
    if any(not np.array_equal(r["step"], steps) for r in runs[1:]):
        raise ValueError("Inconsistent step axes across runs; check that steps/runs settings match.")
    summary: Dict[str, np.ndarray] = {"step": steps}

    for key in METRIC_KEYS:
        stacked = np.stack([r[key] for r in runs], axis=0)
        summary[f"{key}_mean"] = stacked.mean(axis=0)
        summary[f"{key}_std"] = stacked.std(axis=0)

    summary_cols = ["step"]
    for key in METRIC_KEYS:
        summary_cols.append(f"{key}_mean")
        summary_cols.append(f"{key}_std")
    summary_data = np.column_stack([summary[col] for col in summary_cols])
    fmt = ["%d"] + ["%.8f"] * (len(summary_cols) - 1)
    np.savetxt(size_dir / "summary.csv", summary_data, delimiter=",", header=",".join(summary_cols), comments="", fmt=fmt)

    eps = 1e-12
    fmax_runs = [-np.log10(np.clip(np.abs(1.0 - r["max_fitness"]), eps, None)) for r in runs]
    fmean_runs = [-np.log10(np.clip(np.abs(1.0 - r["mean_fitness"]), eps, None)) for r in runs]
    size_max_runs = [r["max_size"] for r in runs]
    size_mean_runs = [r["mean_size"] for r in runs]
    cell_cum_runs = [r["cum_cell_types"] for r in runs]
    pattern_cum_runs = [r["cum_pattern_types"] for r in runs]

    _plot_runs(fmax_runs, size_dir / f"fig4_max_fitness_L{size}.png", ylabel="-log10|1-fitness| (max)", log_time=True)
    _plot_runs(fmean_runs, size_dir / f"fig4_mean_fitness_L{size}.png", ylabel="-log10|1-fitness| (mean)", log_time=True)
    _plot_runs(size_max_runs, size_dir / f"fig5_max_size_L{size}.png", ylabel="Max component size", log_time=True)
    _plot_runs(size_mean_runs, size_dir / f"fig5_mean_size_L{size}.png", ylabel="Mean component size", log_time=True)
    _plot_runs(cell_cum_runs, size_dir / f"fig6_cum_cell_types_L{size}.png", ylabel="Cumulative cell types", log_time=True)
    _plot_runs(pattern_cum_runs, size_dir / f"fig6_cum_pattern_types_L{size}.png", ylabel="Cumulative pattern types", log_time=True)

    print(f"[aggregate] Saved summary and plots for L={size} to {size_dir}")
    return summary


def _load_summary_file(size_dir: Path) -> Dict[str, np.ndarray] | None:
    path = size_dir / "summary.csv"
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        return None

    def _as_array(x):
        arr = np.array(x, copy=False)
        return arr if arr.ndim > 0 else np.array([arr])

    summary = {name: _as_array(data[name]) for name in data.dtype.names}
    return summary


def _collect_available_summaries(output_root: Path) -> List[Tuple[int, Dict[str, np.ndarray]]]:
    summaries: List[Tuple[int, Dict[str, np.ndarray]]] = []
    for path in sorted(output_root.glob("L*/summary.csv")):
        size_str = path.parent.name
        if not size_str.startswith("L"):
            continue
        try:
            size_val = int(size_str[1:])
        except ValueError:
            continue
        summary = _load_summary_file(path.parent)
        if summary is not None:
            summaries.append((size_val, summary))
    return summaries


def _plot_multi_size(size_summaries: List[Tuple[int, Dict[str, np.ndarray]]], outdir: Path) -> None:
    if not size_summaries:
        return
    size_summaries = sorted(size_summaries, key=lambda x: x[0])
    steps = size_summaries[0][1]["step"]
    for _, summary in size_summaries[1:]:
        if not np.array_equal(summary["step"], steps):
            raise ValueError("Inconsistent step axes across sizes; ensure --steps is the same.")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(size_summaries)))

    def make_plot(value_fn, ylabel: str, filename: str, log_time: bool = True):
        plt.figure(figsize=(6, 3))
        for (size, summary), color in zip(size_summaries, colors):
            plt.plot(steps, value_fn(summary), label=f"L={size}", color=color, linewidth=1.6)
        if log_time:
            plt.xscale("log")
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=200)
        plt.close()

    eps = 1e-12
    make_plot(
        lambda s: -np.log10(np.clip(np.abs(1.0 - s["max_fitness_mean"]), eps, None)),
        "-log10|1-fitness| (max)",
        "fig4_max_fitness_all_sizes.png",
    )
    make_plot(
        lambda s: -np.log10(np.clip(np.abs(1.0 - s["mean_fitness_mean"]), eps, None)),
        "-log10|1-fitness| (mean)",
        "fig4_mean_fitness_all_sizes.png",
    )
    make_plot(lambda s: s["max_size_mean"], "Max component size", "fig5_max_size_all_sizes.png")
    make_plot(lambda s: s["mean_size_mean"], "Mean component size", "fig5_mean_size_all_sizes.png")
    make_plot(lambda s: s["cum_cell_types_mean"], "Cumulative cell types", "fig6_cum_cell_types_all_sizes.png")
    make_plot(lambda s: s["cum_pattern_types_mean"], "Cumulative pattern types", "fig6_cum_pattern_types_all_sizes.png")
    print(f"[aggregate] Saved multi-size plots to {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Run SCHC at multiple grid sizes and save numeric metrics + plots.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0, help="Base seed; seeds will be seed0, seed0+1, ...")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--mu", type=str, default="0.002/0.999")
    parser.add_argument("--death-prob", type=float, default=0.001)
    parser.add_argument("--progress-interval", type=int, default=200)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "large_space")
    parser.add_argument("--skip-aggregate", action="store_true", help="Skip summary/plots (useful for split GPU batches).")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not launch new runs; just aggregate existing CSVs under each size directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-seed CSVs.")
    args = parser.parse_args()

    mu_val = float(eval(str(args.mu), {"__builtins__": None}, {}))
    for size in args.sizes:
        params = SCHCParams(k=args.k, n=args.n, L=size, mu=mu_val, death_prob=args.death_prob)
        size_dir = args.output_root / f"L{size}"
        run_dir = size_dir / "runs"
        size_dir.mkdir(parents=True, exist_ok=True)

        if not args.aggregate_only:
            for i in range(args.runs):
                seed = args.seed0 + i
                outpath = run_dir / f"seed_{seed}.csv"
                if outpath.exists() and not args.overwrite:
                    print(f"[skip] {outpath} exists; use --overwrite to recompute.")
                    continue
                print(f"[run] L={size} seed={seed}")
                metrics = run_single(params, args.steps, seed, progress_interval=args.progress_interval)
                _save_run_csv(metrics, run_dir, seed)

        if not args.skip_aggregate:
            _aggregate_and_plot(size_dir, size)

    if not args.skip_aggregate:
        size_summaries = _collect_available_summaries(args.output_root)
        if len(size_summaries) >= 2:
            _plot_multi_size(size_summaries, args.output_root)


if __name__ == "__main__":
    main()
