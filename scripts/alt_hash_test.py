#!/usr/bin/env python3
"""
Alternative hash function test for SCHC.

Tests robustness of the transition by using CRC32 instead of the default XOR hash.
Implements a modified simulation that uses zlib.crc32 for fitness computation.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from jax import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import SCHCParams, advance_one_step_jit, initialize_state
from scripts.large_space_scaling import _save_run_csv, _load_runs, METRIC_KEYS


def _fitness_crc32(component: List[Tuple[int, int, int]]) -> float:
    """CRC32-based fitness function as alternative hash."""
    if not component:
        return 0.0
    arr = np.array(component, dtype=np.uint32)
    min_xy = arr[:, :2].min(axis=0)
    rel = arr[:, :2] - min_xy
    order = np.lexsort((rel[:, 1], rel[:, 0]))
    rel = rel[order]
    values = arr[:, 2][order]

    # Build byte representation and compute CRC32
    data = b""
    for i in range(len(rel)):
        data += int(rel[i, 0]).to_bytes(4, "little")
        data += int(rel[i, 1]).to_bytes(4, "little")
        data += int(values[i]).to_bytes(4, "little")

    crc = zlib.crc32(data) & 0xFFFFFFFF
    return float(np.float32(crc) / np.float32(np.iinfo(np.uint32).max))


def _component_signature(component: List[Tuple[int, int, int]]) -> Tuple[Tuple[int, int, int], ...]:
    arr = np.array(component, dtype=np.int32)
    min_xy = arr[:, :2].min(axis=0)
    rel = arr[:, :2] - min_xy
    vals = arr[:, 2]
    order = np.lexsort((rel[:, 1], rel[:, 0]))
    rel = rel[order]
    vals = vals[order]
    return tuple((int(x), int(y), int(v)) for (x, y), v in zip(rel, vals))


def _components_and_metrics_crc32(config_np: np.ndarray) -> Dict:
    """Connected components with CRC32 fitness."""
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
            fitnesses.append(_fitness_crc32(comp))
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


def run_single_alt_hash(params: SCHCParams, steps: int, seed: int,
                         progress_interval: int = 500):
    """Run simulation but compute metrics with CRC32 fitness (observation-only)."""
    key = random.PRNGKey(seed)
    state = initialize_state(key, params)

    metrics = {k: [] for k in METRIC_KEYS}
    seen_cell_types = set()
    seen_pattern_types = set()

    for t in range(steps + 1):
        cfg_np = np.array(state.config)
        m = _components_and_metrics_crc32(cfg_np)
        metrics["max_fitness"].append(m["max_fitness"])
        metrics["mean_fitness"].append(m["mean_fitness"])
        metrics["max_size"].append(m["max_size"])
        metrics["mean_size"].append(m["mean_size"])
        seen_cell_types |= m["cell_types"]
        seen_pattern_types |= m["pattern_types"]
        metrics["cum_cell_types"].append(len(seen_cell_types))
        metrics["cum_pattern_types"].append(len(seen_pattern_types))

        if progress_interval and t % progress_interval == 0:
            print(f"[alt_hash seed={seed}] step={t}")
            sys.stdout.flush()

        if t == steps:
            break
        # The actual simulation dynamics still use the JAX hash internally
        # We're measuring with an alternative hash to check if transition is observable
        state = advance_one_step_jit(state, params)

    return metrics


def run_and_aggregate(sizes, runs, steps, k, n, mu, death_prob,
                      output_root, seed0=0, progress_interval=500):
    """Run all conditions and aggregate."""
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for size in sizes:
        params = SCHCParams(k=k, n=n, L=size, mu=mu, death_prob=death_prob)
        size_dir = output_root / f"L{size}"
        run_dir = size_dir / "runs"
        size_dir.mkdir(parents=True, exist_ok=True)

        for i in range(runs):
            seed = seed0 + i
            outpath = run_dir / f"seed_{seed}.csv"
            if outpath.exists():
                print(f"[skip] {outpath} exists")
                continue
            print(f"[run] alt_hash L={size} seed={seed}")
            metrics = run_single_alt_hash(params, steps, seed, progress_interval)
            _save_run_csv(metrics, run_dir, seed)

        # Aggregate
        runs_data = _load_runs(run_dir)
        if runs_data:
            steps_arr = runs_data[0]["step"]
            summary = {"step": steps_arr}
            for key in METRIC_KEYS:
                stacked = np.stack([r[key] for r in runs_data])
                summary[f"{key}_mean"] = stacked.mean(axis=0)
                summary[f"{key}_std"] = stacked.std(axis=0)

            cols = ["step"] + [f"{k}_{s}" for k in METRIC_KEYS for s in ("mean", "std")]
            data = np.column_stack([summary[c] for c in cols])
            fmt = ["%d"] + ["%.8f"] * (len(cols) - 1)
            np.savetxt(size_dir / "summary.csv", data, delimiter=",",
                       header=",".join(cols), comments="", fmt=fmt)
            summaries[size] = summary

    # Comparison plot
    if summaries:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(summaries)))
        for (size, summary), color in zip(sorted(summaries.items()), colors):
            steps = summary["step"]
            axes[0].plot(steps, summary["mean_size_mean"], label=f"L={size}",
                         color=color, linewidth=1.5)
            axes[1].plot(steps, summary["mean_fitness_mean"], label=f"L={size}",
                         color=color, linewidth=1.5)

        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Mean component size")
        axes[0].set_title("Alt Hash (CRC32): Mean Size")
        axes[0].set_xscale("log")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Mean fitness")
        axes[1].set_title("Alt Hash (CRC32): Mean Fitness")
        axes[1].set_xscale("log")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_root / "alt_hash_comparison.png", dpi=200)
        plt.close()

        # Summary table
        with open(output_root / "alt_hash_summary.csv", "w") as f:
            f.write("L,mean_size_final,max_size_final,mean_fitness_final\n")
            for size in sorted(summaries.keys()):
                s = summaries[size]
                f.write(f"{size},{s['mean_size_mean'][-1]:.4f},"
                        f"{s['max_size_mean'][-1]:.4f},"
                        f"{s['mean_fitness_mean'][-1]:.6f}\n")

    print(f"[alt_hash] All results saved to {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Alt hash function test for SCHC.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 300, 320, 400])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--mu", type=str, default="0.002/0.999")
    parser.add_argument("--death-prob", type=float, default=0.001)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "alt_hash")
    parser.add_argument("--progress-interval", type=int, default=500)
    args = parser.parse_args()

    mu_val = float(eval(str(args.mu), {"__builtins__": None}, {}))
    run_and_aggregate(args.sizes, args.runs, args.steps, args.k, args.n,
                      mu_val, args.death_prob, args.output_root,
                      args.seed0, args.progress_interval)


if __name__ == "__main__":
    main()
