"""
Reproduce key analyses from arXiv:2412.12790v1 for Structural Cellular Hash Chemistry.

Runs a single SCHC simulation, collects time series metrics, and saves plots into `results/`.
This is a practical approximation of the Mathematica notebooks in Python/JAX.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import SCHCParams, advance_one_step_jit, initialize_state


def _fitness_numpy(component: List[Tuple[int, int, int]]) -> float:
    """Compute the same hash-based fitness used in the simulator, in NumPy."""

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
    mixed = mixed & np.uint64(0xFFFFFFFF)  # keep within 32 bits to mirror uint32 wraparound
    return float(mixed.astype(np.float32) / np.float32(np.iinfo(np.uint32).max))


def _components_and_metrics(config_np: np.ndarray) -> Dict[str, float]:
    """Find connected components (8-neighborhood) on CPU and derive summary metrics."""

    L = config_np.shape[0]
    visited = np.zeros_like(config_np, dtype=bool)
    components: List[List[Tuple[int, int, int]]] = []
    sizes: List[int] = []
    fitnesses: List[float] = []
    num_active = int((config_np > 0).sum())

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for x in range(L):
        for y in range(L):
            if config_np[x, y] == 0 or visited[x, y]:
                continue
            # BFS
            stack = [(x, y)]
            visited[x, y] = True
            comp: List[Tuple[int, int, int]] = []
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy, int(config_np[cx, cy])))
                for dx, dy in neighbors:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < L and 0 <= ny < L and config_np[nx, ny] != 0 and not visited[nx, ny]:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            components.append(comp)
            sizes.append(len(comp))
            fitnesses.append(_fitness_numpy(comp))

    if not components:
        return {
            "active_cells": 0,
            "num_components": 0,
            "mean_size": 0.0,
            "max_size": 0.0,
            "mean_fitness": 0.0,
            "max_fitness": 0.0,
        }

    fitness_arr = np.array(fitnesses, dtype=np.float32)
    size_arr = np.array(sizes, dtype=np.float32)
    return {
        "active_cells": num_active,
        "num_components": len(components),
        "mean_size": float(size_arr.mean()),
        "max_size": float(size_arr.max()),
        "mean_fitness": float(fitness_arr.mean()),
        "max_fitness": float(fitness_arr.max()),
    }


def run_simulation(
    params: SCHCParams, steps: int, snapshot_times: List[int], seed: int, progress_interval: int
) -> Dict[str, np.ndarray]:
    key = random.PRNGKey(seed)
    state = initialize_state(key, params)

    records = []
    snapshots: Dict[int, np.ndarray] = {}

    for t in range(steps + 1):
        config_np = np.array(state.config)
        metrics = _components_and_metrics(config_np)
        metrics["step"] = t
        records.append(metrics)
        if t in snapshot_times:
            snapshots[t] = config_np.copy()
        if progress_interval and t % progress_interval == 0:
            print(
                f"[progress] step={t} active={metrics['active_cells']} max_size={metrics['max_size']:.2f}",
                flush=True,
            )

        if t == steps:
            break
        state = advance_one_step_jit(state, params)

    return {"records": records, "snapshots": snapshots}


def save_plots(records: List[Dict[str, float]], snapshots: Dict[int, np.ndarray], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    steps = [r["step"] for r in records]
    for key in ["max_fitness", "mean_fitness", "max_size", "mean_size", "active_cells", "num_components"]:
        vals = [r[key] for r in records]
        plt.figure(figsize=(6, 3))
        plt.plot(steps, vals, label=key)
        plt.xlabel("Step")
        plt.ylabel(key.replace("_", " "))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"{key}.png", dpi=200)
        plt.close()

    if snapshots:
        cols = len(snapshots)
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3), squeeze=False)
        for idx, (t, img) in enumerate(sorted(snapshots.items())):
            ax = axes[0, idx]
            ax.imshow(img, cmap="hsv", interpolation="nearest")
            ax.set_title(f"t={t}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(outdir / "snapshots.png", dpi=200)
        plt.close()


def save_csv(records: List[Dict[str, float]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    keys = ["step", "active_cells", "num_components", "mean_size", "max_size", "mean_fitness", "max_fitness"]
    header = ",".join(keys)
    lines = [header]
    for r in records:
        lines.append(",".join(str(r[k]) for k in keys))
    (outdir / "metrics.csv").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce analyses from the SCHC paper (arXiv:2412.12790v1).")
    parser.add_argument("--steps", type=int, default=500, help="Number of onestep iterations (paper used 2000).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--size", type=int, default=100, help="Grid size L.")
    parser.add_argument("--k", type=int, default=1000, help="Number of possible cell types.")
    parser.add_argument("--n", type=int, default=10, help="Initial number of active cells.")
    parser.add_argument("--mu", type=float, default=0.002 / 0.999, help="Mutation probability.")
    parser.add_argument("--death-prob", type=float, default=0.001, help="Death probability when copying.")
    parser.add_argument(
        "--snapshots",
        type=str,
        default="0,100,250,500",
        help="Comma-separated step indices to save snapshots.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print progress every N steps (set 0 to disable).",
    )
    args = parser.parse_args()

    params = SCHCParams(k=args.k, n=args.n, L=args.size, mu=args.mu, death_prob=args.death_prob)
    snapshot_times = [int(s) for s in args.snapshots.split(",") if s]

    outdir = Path(__file__).resolve().parent
    results = run_simulation(params, args.steps, snapshot_times, args.seed, args.progress_interval)
    records = results["records"]
    snapshots = results["snapshots"]

    save_csv(records, outdir)
    save_plots(records, snapshots, outdir)

    (outdir / "params.txt").write_text(str(asdict(params)))
    print(f"Saved metrics and plots to {outdir}")


if __name__ == "__main__":
    main()
