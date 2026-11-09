#!/usr/bin/env python3
"""
Run the JAX SCHC in the original setting (L=100, n=10, k=1000, 2000 steps) and
record the per-step statistics of Sayama's Mathematica analysis
(SCHC-systematic-sim-data-analysis.nb), so that the runs can be compared
quantitatively with the aggregated data files of the original study
(github.com/hsayama/Structural-Cellular-Hash-Chemistry, *Series.dat).

Per step t (0-based, as the Mathematica `time`), over the contest winners of the step:
  max_fitness, mean_fitness       maxFitnessSeries / meanFitnessSeries
  num_replicated_cells            numIndivEntitiesSeries (sum of winner sizes)
  max_rep_size, mean_rep_size     maxRepEventSeries / meanRepEventSeries
  cum_cell_types                  nptypesSeries (distinct cell types in winner patterns)
  cum_pattern_types               nctypesSeries (distinct winner patterns)
plus the gaps 1 - fitness (full precision), n_events, n_active and the cumulative
size of the Mathematica log file this run would have written (the original
analysis keeps only runs whose log exceeds 1 MB).

Seeds are batched with vmap on one GPU; each arm is a (winner_rule, score) pair.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation import SCHCParams, advance_one_step_logged, initialize_state
from scripts.run_provenance import FIXED_ROOT, register_run

ARMS = {
    "legacy": dict(winner_rule="legacy", score="xor"),      # published JAX model
    "fixed": dict(winner_rule="original", score="xor"),     # Mathematica winner rule
    "fixed_mix": dict(winner_rule="original", score="mix"), # + non-degenerate score
}
UINT_MAX = float(0xFFFFFFFF)
HEADER = ("step,max_fitness,mean_fitness,max_fitness_gap,mean_fitness_gap,num_replicated_cells,"
          "max_rep_size,mean_rep_size,cum_cell_types,cum_pattern_types,n_events,n_active,log_bytes_cum")


def run_batch_reference(params: SCHCParams, seeds, steps: int):
    max_events = params.L * params.L + 1
    init = jax.jit(jax.vmap(lambda key: initialize_state(key, params)))
    step = jax.jit(jax.vmap(lambda s, seen: advance_one_step_logged(s, params, seen, max_events)))

    states = init(jnp.stack([random.PRNGKey(s) for s in seeds]))
    seen = jnp.zeros((len(seeds), params.k + 1), dtype=bool)
    patterns = [set() for _ in seeds]
    rows = [[] for _ in seeds]
    log_bytes = np.zeros(len(seeds), dtype=np.int64)
    for t in range(steps):
        states, seen, st = step(states, seen)
        st = jax.device_get(st)
        n_types = np.asarray(seen[:, 1:].sum(axis=1))
        for b in range(len(seeds)):
            n = int(st["n_events"][b])
            if n > max_events:
                raise RuntimeError(f"event buffer overflow: seed {seeds[b]} step {t} n={n}")
            ids = st["ids"][b, :n].astype(np.uint64)
            patterns[b].update(((ids[:, 0] << np.uint64(32)) | ids[:, 1]).tolist())
            log_bytes[b] += int(st["log_bytes"][b])
            if n > 0:
                max_gap = (UINT_MAX - float(st["max_hash"][b])) / UINT_MAX
                mean_gap = float(st["sum_gap"][b]) / n
                vals = (1.0 - max_gap, 1.0 - mean_gap, max_gap, mean_gap, int(st["sum_size"][b]),
                        int(st["max_size"][b]), int(st["sum_size"][b]) / n)
            else:  # extinct: no replication events in this step
                vals = (np.nan,) * 4 + (0, 0, np.nan)
            rows[b].append((t, *vals, int(n_types[b]), len(patterns[b]), n,
                            int(st["n_active"][b]), int(log_bytes[b])))
    return rows


def run_batch(params: SCHCParams, seeds, steps: int):
    from scripts.adaptive_batch import run_batch as adaptive_run
    return adaptive_run(params, seeds, steps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--seeds", default="0-99", help="e.g. 0-99 or 0,1,2")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--output-root", default=str(FIXED_ROOT / "original_comparison"))
    ap.add_argument("--legacy-corner", action="store_true", help="reproduce the archived corner-write bug")
    args = ap.parse_args()

    if "-" in args.seeds:
        lo, hi = map(int, args.seeds.split("-"))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
    params = SCHCParams(L=args.size, legacy_corner=args.legacy_corner, **ARMS[args.arm])
    out_dir = Path(args.output_root) / args.arm
    register_run(out_dir, params, seeds, args.steps)
    print(f"arm={args.arm} params={params} seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) "
          f"devices={jax.devices()}", flush=True)

    fmt = ["%d"] + ["%.10g"] * 7 + ["%d"] * 5
    for i in range(0, len(seeds), args.batch):
        batch = seeds[i:i + args.batch]
        if all((out_dir / f"seed_{s}.csv").exists() for s in batch):
            continue
        t0 = time.time()
        rows = run_batch(params, batch, args.steps)
        for s, r in zip(batch, rows):
            np.savetxt(out_dir / f"seed_{s}.csv", np.array(r, dtype=np.float64), delimiter=",",
                       header=HEADER, comments="", fmt=fmt)
        print(f"seeds {batch[0]}..{batch[-1]} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
