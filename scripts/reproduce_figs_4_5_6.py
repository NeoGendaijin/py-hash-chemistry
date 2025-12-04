"""
Approximate reproduction of Figures 4, 5, 6 from arXiv:2412.12790v1.

For multiple seeds, run SCHC simulations, compute component-level metrics per step,
and plot red individual runs + black averages:
 - Fig4: max/mean fitness (using -log10|1-fitness|)
 - Fig5: max/mean component size
 - Fig6: cumulative unique cell types and pattern types

This uses the JAX simulator for stepping, but metrics are computed on CPU/NumPy
from full grids each step. This is an approximation of the Mathematica analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import SCHCParams, advance_one_step_jit, initialize_state


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


def main():
    parser = argparse.ArgumentParser(description="Approximate reproduction of Figures 4, 5, 6.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0, help="Base seed; seeds will be seed0, seed0+1, ...")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--mu", type=str, default="0.002/0.999")
    parser.add_argument("--death-prob", type=float, default=0.001)
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()

    mu_val = float(eval(str(args.mu), {"__builtins__": None}, {}))
    params = SCHCParams(k=args.k, n=args.n, L=args.size, mu=mu_val, death_prob=args.death_prob)
    outdir = Path(__file__).resolve().parent / "figs_4_5_6"
    outdir.mkdir(parents=True, exist_ok=True)

    fitness_max_runs = []
    fitness_mean_runs = []
    size_max_runs = []
    size_mean_runs = []
    cell_cum_runs = []
    pattern_cum_runs = []

    for i in range(args.runs):
        seed = args.seed0 + i
        print(f"[run] seed={seed}")
        metrics = run_single(params, args.steps, seed, progress_interval=args.progress_interval)
        # transform fitness to -log10|1 - fitness|
        eps = 1e-12
        fmax = -np.log10(np.clip(np.abs(1.0 - np.array(metrics["max_fitness"])), eps, None))
        fmean = -np.log10(np.clip(np.abs(1.0 - np.array(metrics["mean_fitness"])), eps, None))
        fitness_max_runs.append(fmax)
        fitness_mean_runs.append(fmean)
        size_max_runs.append(np.array(metrics["max_size"]))
        size_mean_runs.append(np.array(metrics["mean_size"]))
        cell_cum_runs.append(np.array(metrics["cum_cell_types"]))
        pattern_cum_runs.append(np.array(metrics["cum_pattern_types"]))

    _plot_runs(fitness_max_runs, outdir / "fig4_max_fitness.png", ylabel="-log10|1-fitness| (max)", log_time=True)
    _plot_runs(fitness_mean_runs, outdir / "fig4_mean_fitness.png", ylabel="-log10|1-fitness| (mean)", log_time=True)
    _plot_runs(size_max_runs, outdir / "fig5_max_size.png", ylabel="Max component size", log_time=True)
    _plot_runs(size_mean_runs, outdir / "fig5_mean_size.png", ylabel="Mean component size", log_time=True)
    _plot_runs(cell_cum_runs, outdir / "fig6_cum_cell_types.png", ylabel="Cumulative cell types", log_time=True)
    _plot_runs(pattern_cum_runs, outdir / "fig6_cum_pattern_types.png", ylabel="Cumulative pattern types", log_time=True)

    print(f"Saved figure approximations to {outdir}")


if __name__ == "__main__":
    main()
