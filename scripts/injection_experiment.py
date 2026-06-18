#!/usr/bin/env python3
"""Transplantation (injection) experiment.

Takes a fully developed runaway structure from a saved configuration, places
its largest connected component (centered, cropping anything that does not fit)
into otherwise-empty grids of varying size L, runs the dynamics, and records
whether the structure takes over or is contained/dissolved as a function of L.

Tests the finite-size-container hypothesis: a structure that dominates a large
grid should be contained when transplanted into a sufficiently small one.

Example:
  python scripts/injection_experiment.py \
      --source results/gifs/configs_L400_seed8/config_step10000.npy \
      --sizes 100 150 200 250 300 350 400 --seeds 0-4 --steps 10000 \
      --output-root results/injection
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
from jax import random
import jax.numpy as jnp
from src import SCHCParams, advance_one_step_jit, initialize_state
from src.simulation import SCHCState

RUNAWAY_THRESHOLD = 100.0


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def largest_component_cells(config_np):
    """Return (rel_xy, types) of the largest 8-connected component, centered at its bbox center."""
    occ = config_np > 0
    labeled, n = ndimage.label(occ, structure=np.ones((3, 3)))
    sizes = ndimage.sum(occ, labeled, range(1, n + 1))
    lab = int(np.argmax(sizes)) + 1
    xs, ys = np.where(labeled == lab)
    types = config_np[xs, ys]
    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    rel = np.stack([xs - cx, ys - cy], axis=1)  # centered, float
    return rel, types, (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)


def inject(rel, types, L):
    """Place the (centered) structure into an empty L x L grid; crop cells that fall outside."""
    grid = np.zeros((L, L), dtype=np.int32)
    cx = cy = L // 2
    tx = np.rint(rel[:, 0] + cx).astype(int)
    ty = np.rint(rel[:, 1] + cy).astype(int)
    inb = (tx >= 0) & (tx < L) & (ty >= 0) & (ty < L)
    grid[tx[inb], ty[inb]] = types[inb]
    return grid, int(inb.sum())


def metrics(config_np):
    occ = config_np > 0
    n_active = int(occ.sum())
    if n_active == 0:
        return 0, 0, 0.0
    labeled, n = ndimage.label(occ, structure=np.ones((3, 3)))
    sizes = ndimage.sum(occ, labeled, range(1, n + 1))
    max_comp = int(sizes.max())
    mean_comp = float(np.mean(sizes))
    return n_active, max_comp, mean_comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path,
                    default=ROOT / "results/gifs/configs_L400_seed8/config_step10000.npy")
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=str, default="0-4")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--mu", type=str, default="0.002/0.999")
    ap.add_argument("--death-prob", type=float, default=0.001)
    ap.add_argument("--output-root", type=Path, default=ROOT / "results/injection")
    args = ap.parse_args()

    mu_val = float(eval(str(args.mu), {"__builtins__": None}, {}))
    seeds = parse_seeds(args.seeds)
    src = np.load(args.source)
    rel, types, extent = largest_component_cells(src)
    print(f"[source] {args.source.name}: largest component {len(types)} cells, "
          f"bbox extent {extent}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for L in args.sizes:
        init_grid, n_injected = inject(rel, types, L)
        for seed in seeds:
            params = SCHCParams(k=args.k, n=10, L=L, mu=mu_val,
                                death_prob=args.death_prob, periodic=False)
            base = initialize_state(random.PRNGKey(seed), params)
            state = SCHCState(config=jnp.asarray(init_grid), rng=base.rng,
                              time=jnp.array(0, dtype=jnp.int32))
            for _ in range(args.steps):
                state = advance_one_step_jit(state, params)
            cfg = np.array(state.config)
            n_active, max_comp, mean_comp = metrics(cfg)
            took_over = max_comp > 0.3 * L * L
            rows.append((L, seed, n_injected, n_active, max_comp, mean_comp,
                         max_comp / (L * L), int(took_over)))
            print(f"[L={L} seed={seed}] injected={n_injected} -> active={n_active} "
                  f"max_comp={max_comp} ({max_comp/(L*L):.2f} of L^2) "
                  f"{'TAKEOVER' if took_over else 'contained'}", flush=True)

    import csv
    out = args.output_root / "injection_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L", "seed", "n_injected", "n_active_final", "max_comp_final",
                    "mean_comp_final", "max_comp_frac_L2", "takeover"])
        w.writerows(rows)
    print(f"\n[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
