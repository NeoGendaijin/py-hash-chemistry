#!/usr/bin/env python3
"""
(D, P) sweep of spatial-dyadic SCHC in the setting of Sayama's ALIFE 2026 LBA and
npj Extension I: L=100, n=10, k=1000, 2000 steps, D in {10, ..., 100}, P in {0, 0.2, ..., 1}.

Uses the JAX model validated against the original Mathematica runs
(winner_rule="original", score="mix"; see scripts/compare_original_schc.py) and
records the same per-step statistics. dyadic_rule selects the intended dyadic
contest or the behaviour of the Mathematica code (the order swap never runs).
P=0 conditions do not depend on dyadic_rule; they are run once, under "intended".

Output: <output-root>/<dyadic_rule>/D{D}_P{P}/seed_{s}.csv (existing complete
conditions are skipped, so the sweep can be resumed).

Example (one process per GPU, conditions split across processes):
  CUDA_VISIBLE_DEVICES=0 python scripts/run_spatial_dyadic_sweep.py --part 0/3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_original_schc import HEADER, run_batch
from src.simulation import SCHCParams
from scripts.run_provenance import FIXED_ROOT, register_run

D_VALUES = list(range(10, 101, 10))
P_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Corners first, so that a partial sweep already shows the main effects.
PRIORITY_D = [10, 50, 100]
PRIORITY_P = [0.0, 0.2, 0.6, 1.0]


def conditions(rules):
    """(rule, D, P) in priority order, rules interleaved; P=0 only under "intended"."""
    first = [(d, p) for p in PRIORITY_P for d in PRIORITY_D]
    rest = [(d, p) for p in P_VALUES for d in D_VALUES if (d, p) not in first]
    return [(r, d, p) for d, p in first + rest for r in rules if not (p == 0.0 and r == "as_coded")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dyadic-rules", nargs="+", choices=["intended", "as_coded"], default=["intended", "as_coded"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--part", default="0/1", help="i/n: run every n-th condition starting at i")
    ap.add_argument("--output-root", default=str(FIXED_ROOT / "spatial_dyadic"))
    ap.add_argument("--legacy-corner", action="store_true", help="reproduce the archived corner-write bug")
    args = ap.parse_args()

    i, n = map(int, args.part.split("/"))
    todo = conditions(args.dyadic_rules)[i::n]
    seeds = list(range(args.seeds))
    fmt = ["%d"] + ["%.10g"] * 7 + ["%d"] * 5
    print(f"rules={args.dyadic_rules} part={args.part}: {len(todo)} conditions", flush=True)
    for rule, d, p in todo:
        out_dir = Path(args.output_root) / rule / f"D{d}_P{p:.1f}"
        params = SCHCParams(L=100, winner_rule="original", score="mix", legacy_corner=args.legacy_corner,
                            D=d, P=p, dyadic_rule=rule)
        register_run(out_dir, params, seeds, args.steps)
        if all((out_dir / f"seed_{s}.csv").exists() for s in seeds):
            print(f"skip {rule} D={d} P={p}", flush=True)
            continue
        t0 = time.time()
        rows = run_batch(params, seeds, args.steps)
        out_dir.mkdir(parents=True, exist_ok=True)
        for s, r in zip(seeds, rows):
            np.savetxt(out_dir / f"seed_{s}.csv", np.array(r, dtype=np.float64), delimiter=",",
                       header=HEADER, comments="", fmt=fmt)
        print(f"{rule} D={d} P={p} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
