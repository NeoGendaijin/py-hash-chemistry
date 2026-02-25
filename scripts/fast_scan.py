#!/usr/bin/env python3
"""
Fast version of large_space_scaling.py that subsamples metrics.

Instead of computing connected components every step (expensive for L>=300),
this version only computes metrics every `sample_interval` steps, using
`run_steps_jit` for the simulation steps in between.

Output format is compatible with the existing CSV format but with fewer rows.
The output CSV contains only sampled rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from jax import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import SCHCParams, advance_one_step_jit, initialize_state


METRIC_KEYS = ["max_fitness", "mean_fitness", "max_size", "mean_size",
               "cum_cell_types", "cum_pattern_types"]


def _fitness_numpy(component: List[Tuple[int, int, int]]) -> float:
    if not component:
        return 0.0
    arr = np.array(component, dtype=np.uint32)
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


def _component_signature(component):
    arr = np.array(component, dtype=np.int32)
    min_xy = arr[:, :2].min(axis=0)
    rel = arr[:, :2] - min_xy
    vals = arr[:, 2]
    order = np.lexsort((rel[:, 1], rel[:, 0]))
    rel = rel[order]
    vals = vals[order]
    return tuple((int(x), int(y), int(v)) for (x, y), v in zip(rel, vals))


def _components_and_metrics(config_np: np.ndarray):
    L = config_np.shape[0]
    visited = np.zeros_like(config_np, dtype=bool)
    sizes = []
    fitnesses = []
    cell_types = []
    pattern_signatures = []

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for x in range(L):
        for y in range(L):
            if config_np[x, y] == 0 or visited[x, y]:
                continue
            stack = [(x, y)]
            visited[x, y] = True
            comp = []
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
            "mean_size": 0.0, "max_size": 0.0,
            "mean_fitness": 0.0, "max_fitness": 0.0,
            "cell_types": set(), "pattern_types": set(),
        }
    return {
        "mean_size": float(np.mean(sizes)),
        "max_size": float(np.max(sizes)),
        "mean_fitness": float(np.mean(fitnesses)),
        "max_fitness": float(np.max(fitnesses)),
        "cell_types": set(cell_types),
        "pattern_types": set(pattern_signatures),
    }


def run_single_sampled(params: SCHCParams, steps: int, seed: int,
                        sample_interval: int = 1,
                        progress_interval: int = 0):
    """Run simulation, collecting metrics every sample_interval steps."""
    key = random.PRNGKey(seed)
    state = initialize_state(key, params)

    metrics = {k: [] for k in METRIC_KEYS}
    sampled_steps = []
    seen_cell_types: Set[int] = set()
    seen_pattern_types: Set[Tuple] = set()

    for t in range(steps + 1):
        if t % sample_interval == 0 or t == steps:
            cfg_np = np.array(state.config)
            m = _components_and_metrics(cfg_np)
            metrics["max_fitness"].append(m["max_fitness"])
            metrics["mean_fitness"].append(m["mean_fitness"])
            metrics["max_size"].append(m["max_size"])
            metrics["mean_size"].append(m["mean_size"])
            seen_cell_types |= m["cell_types"]
            seen_pattern_types |= m["pattern_types"]
            metrics["cum_cell_types"].append(len(seen_cell_types))
            metrics["cum_pattern_types"].append(len(seen_pattern_types))
            sampled_steps.append(t)

            if progress_interval and t % progress_interval == 0:
                print(f"[run seed={seed}] step={t} mean_size={m['mean_size']:.1f}")
                sys.stdout.flush()

        if t == steps:
            break
        state = advance_one_step_jit(state, params)

    return metrics, sampled_steps


def save_run_csv(metrics, sampled_steps, run_dir: Path, seed: int):
    run_dir.mkdir(parents=True, exist_ok=True)
    steps_arr = np.array(sampled_steps, dtype=np.int32)
    data = np.column_stack([
        steps_arr,
        np.array(metrics["max_fitness"], dtype=np.float64),
        np.array(metrics["mean_fitness"], dtype=np.float64),
        np.array(metrics["max_size"], dtype=np.float64),
        np.array(metrics["mean_size"], dtype=np.float64),
        np.array(metrics["cum_cell_types"], dtype=np.int64),
        np.array(metrics["cum_pattern_types"], dtype=np.int64),
    ])
    outpath = run_dir / f"seed_{seed}.csv"
    header = "step,max_fitness,mean_fitness,max_size,mean_size,cum_cell_types,cum_pattern_types"
    fmt = ["%d", "%.8f", "%.8f", "%.8f", "%.8f", "%d", "%d"]
    np.savetxt(outpath, data, delimiter=",", header=header, comments="", fmt=fmt)
    return outpath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--sample-interval", type=int, default=10,
                        help="Compute metrics every N steps (1=every step)")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--mu", type=str, default="0.002/0.999")
    parser.add_argument("--death-prob", type=float, default=0.001)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=1000)
    args = parser.parse_args()

    mu_val = float(eval(str(args.mu), {"__builtins__": None}, {}))
    params = SCHCParams(k=args.k, n=args.n, L=args.size, mu=mu_val, death_prob=args.death_prob)

    run_dir = args.output_root / f"L{args.size}" / "runs"
    outpath = run_dir / f"seed_{args.seed}.csv"
    if outpath.exists():
        print(f"[skip] {outpath} exists")
        return

    print(f"[run] L={args.size} seed={args.seed} sample_interval={args.sample_interval}")
    metrics, sampled_steps = run_single_sampled(
        params, args.steps, args.seed,
        sample_interval=args.sample_interval,
        progress_interval=args.progress_interval
    )
    save_run_csv(metrics, sampled_steps, run_dir, args.seed)
    print(f"[done] L={args.size} seed={args.seed}: {len(sampled_steps)} samples saved")


if __name__ == "__main__":
    main()
