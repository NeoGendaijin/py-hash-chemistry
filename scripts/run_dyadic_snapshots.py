#!/usr/bin/env python3
"""
Configurations of spatial-dyadic SCHC runs (L=100) for snapshot figures (npj Fig. 7).

Same model and seeds as scripts/run_spatial_dyadic_sweep.py (winner_rule="original",
score="mix", JAX PRNGKey(seed)), so a saved configuration belongs to the same trajectory
as the corresponding per-step CSV of the sweep; the number of active cells is compared
with that CSV as a check.

Output: <output-root>/<rule>/D{D}_P{P}/seed_{s}_t{t}.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation import SCHCParams, advance_one_step, initialize_state


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", default=["100,0.0", "10,0.0", "100,1.0", "20,0.2"], help="D,P pairs")
    ap.add_argument("--rule", default="intended", choices=["intended", "as_coded"])
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--save-steps", type=int, nargs="+", default=[1500, 1750, 2000])
    ap.add_argument("--output-root", default=str(ROOT / "results" / "spatial_dyadic_snapshots"))
    ap.add_argument("--sweep-root", default=str(ROOT / "results" / "spatial_dyadic"))
    args = ap.parse_args()
    lo, hi = map(int, args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))

    for cond in args.conditions:
        D, P = int(cond.split(",")[0]), float(cond.split(",")[1])
        params = SCHCParams(L=100, winner_rule="original", score="mix", legacy_corner=True,
                            D=D, P=P, dyadic_rule=args.rule)
        step = jax.jit(jax.vmap(lambda s: advance_one_step(s, params)))
        states = jax.jit(jax.vmap(lambda k: initialize_state(k, params)))(jnp.stack([random.PRNGKey(s) for s in seeds]))
        out = Path(args.output_root) / args.rule / f"D{D}_P{P:.1f}"
        out.mkdir(parents=True, exist_ok=True)
        sweep_dir = Path(args.sweep_root) / ("intended" if P == 0.0 else args.rule) / f"D{D}_P{P:.1f}"
        mismatches = 0
        for t in range(1, args.steps + 1):
            states = step(states)
            if t in args.save_steps:
                cfgs = np.asarray(states.config)
                for b, s in enumerate(seeds):
                    np.save(out / f"seed_{s}_t{t}.npy", cfgs[b].astype(np.int16))
                    csv = sweep_dir / f"seed_{s}.csv"
                    if csv.exists():
                        d = pd.read_csv(csv)
                        # the sweep CSV row t records n_active at the start of step t, i.e. after t steps
                        ref = d.loc[d.step == t, "n_active"]
                        if len(ref) and int(ref.iloc[0]) != int((cfgs[b] > 0).sum()):
                            mismatches += 1
        print(f"D={D} P={P}: saved seeds {seeds} at {args.save_steps}; n_active mismatches vs sweep: {mismatches}", flush=True)


if __name__ == "__main__":
    main()
