#!/usr/bin/env python3
"""
Batched (vmap) SCHC size scan for one grid size L, with device-side component metrics.

For the npj Sec. 4 re-run with the corrected simulator (default winner_rule="original",
score="mix"). All seeds of one L run as one batch on one GPU. Every --sample-interval steps
the component metrics (src/metrics.py) are recorded per seed. A seed stops once its mean
component size reaches --stop-mean-size (runaway; same rule as the periodic control):
its configuration is saved and zeroed on the device, which makes its later steps free.

Output (fast_scan-compatible columns plus n_active, n_components):
  <output-root>/L{L}/runs/seed_{s}.csv         written when the seed stops or at the end
  <output-root>/L{L}/runs/seed_{s}.partial.csv  checkpoint every --checkpoint-interval steps
  <output-root>/L{L}/configs/seed_{s}_step{t}.npy  final configuration

Run next to scripts/thermal_watchdog.py (an earlier L=400 scan overheated this machine).
"""

from __future__ import annotations

import argparse
import fcntl
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

from src.metrics import component_metrics
from src.simulation import SCHCParams, SCHCState, advance_one_step, initialize_state
from scripts.run_provenance import FIXED_ROOT, register_run

HEADER = "step,max_fitness,mean_fitness,max_size,mean_size,cum_cell_types,cum_pattern_types,n_active,n_components"
FMT = ["%d", "%.8f", "%.8f", "%d", "%.8f", "%d", "%d", "%d", "%d"]


def parse_seeds(s: str):
    if "-" in s:
        lo, hi = map(int, s.split("-"))
        return list(range(lo, hi + 1))
    return [int(x) for x in s.split(",")]


def save(path: Path, rows):
    np.savetxt(path, np.array(rows, dtype=np.float64), delimiter=",", header=HEADER, comments="", fmt=FMT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--sample-interval", type=int, default=10)
    ap.add_argument("--stop-mean-size", type=float, default=2000.0)
    ap.add_argument("--winner-rule", default="original", choices=["legacy", "original"])
    ap.add_argument("--score", default="mix", choices=["xor", "mix"])
    ap.add_argument("--mu", type=str, default="0.002/0.999")
    ap.add_argument("--death-prob", type=float, default=0.001)
    ap.add_argument("--checkpoint-interval", type=int, default=1000)
    ap.add_argument("--progress-interval", type=int, default=500)
    ap.add_argument("--output-root", default=str(FIXED_ROOT / "transition_fixed"))
    ap.add_argument("--legacy-corner", action="store_true", help="reproduce the archived corner-write bug")
    ap.add_argument("--init-config", default=None, help=".npy configuration shared by all seeds (seeds then only set the RNG)")
    args = ap.parse_args()

    L, seeds = args.size, parse_seeds(args.seeds)
    mu = float(eval(args.mu, {"__builtins__": {}}))
    params = SCHCParams(L=L, mu=mu, death_prob=args.death_prob, winner_rule=args.winner_rule,
                        score=args.score, legacy_corner=args.legacy_corner)
    run_dir = Path(args.output_root) / f"L{L}" / "runs"
    cfg_dir = Path(args.output_root) / f"L{L}" / "configs"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_dir / ".run.lock").open("a+")
    fcntl.flock(lock_handle, fcntl.LOCK_EX)
    register_run(run_dir, params, seeds, args.steps, sample_interval=args.sample_interval,
                 stop_mean_size=args.stop_mean_size, init_config=args.init_config)
    complete = True
    for seed in seeds:
        path = run_dir / f"seed_{seed}.csv"
        if not path.exists():
            complete = False
            break
        last = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)[-1]
        if not (last["step"] == args.steps or last["n_active"] == 0 or
                (args.stop_mean_size > 0 and last["mean_size"] >= args.stop_mean_size)):
            complete = False
            break
    if complete:
        print(f"L={L}: all {len(seeds)} runs already complete; reusing checked trajectories", flush=True)
        return
    cfg_dir.mkdir(parents=True, exist_ok=True)
    max_comp = (L // 2 + 1) ** 2
    print(f"L={L} seeds={seeds} params={params} steps={args.steps} stop_mean_size={args.stop_mean_size} "
          f"devices={jax.devices()}", flush=True)

    step_b = jax.jit(jax.vmap(lambda s: advance_one_step(s, params)))
    metrics_b = jax.jit(jax.vmap(lambda c: component_metrics(c, params.k, max_comp)))
    keys = jnp.stack([random.PRNGKey(s) for s in seeds])
    if args.init_config:
        cfg0 = jnp.asarray(np.load(args.init_config).astype(np.int32))
        states = SCHCState(config=jnp.broadcast_to(cfg0, (len(seeds), L, L)), rng=keys,
                           time=jnp.zeros(len(seeds), dtype=jnp.int32))
    else:
        states = jax.jit(jax.vmap(lambda key: initialize_state(key, params)))(keys)

    B = len(seeds)
    active = np.ones(B, dtype=bool)
    rows = [[] for _ in range(B)]
    types_seen = [set() for _ in range(B)]
    patterns = [set() for _ in range(B)]
    t0 = time.time()
    for t in range(args.steps + 1):
        if t % args.sample_interval == 0 or t == args.steps:
            m = jax.device_get(metrics_b(states.config))
            stop = np.zeros(B, dtype=bool)
            for b in np.flatnonzero(active):
                n = int(m["n_components"][b])
                types_seen[b].update(np.flatnonzero(m["types"][b][1:]) + 1)
                ids = m["pattern_ids"][b, :n].astype(np.uint64)
                patterns[b].update(((ids[:, 0] << np.uint64(32)) | ids[:, 1]).tolist())
                rows[b].append((t, m["max_fitness"][b], m["mean_fitness"][b], m["max_size"][b], m["mean_size"][b],
                                len(types_seen[b]), len(patterns[b]), m["n_active"][b], n))
                if (args.stop_mean_size > 0 and m["mean_size"][b] >= args.stop_mean_size) or m["n_active"][b] == 0 or t == args.steps:
                    stop[b] = True
            if stop.any():
                cfgs = np.asarray(states.config)
                for b in np.flatnonzero(stop):
                    save(run_dir / f"seed_{seeds[b]}.csv", rows[b])
                    np.save(cfg_dir / f"seed_{seeds[b]}_step{t}.npy", cfgs[b].astype(np.int16))
                    (run_dir / f"seed_{seeds[b]}.partial.csv").unlink(missing_ok=True)
                    why = "runaway" if args.stop_mean_size > 0 and rows[b][-1][4] >= args.stop_mean_size else "extinct" if rows[b][-1][7] == 0 else "end"
                    print(f"[L={L} seed={seeds[b]}] stop at step {t} ({why}; mean_size={rows[b][-1][4]:.1f}, "
                          f"max_size={int(rows[b][-1][3])}) after {time.time() - t0:.0f}s", flush=True)
                active &= ~stop
                mask = jnp.asarray(~active)[:, None, None]
                states = SCHCState(config=jnp.where(mask, 0, states.config), rng=states.rng, time=states.time)
            if not active.any():
                break
            if args.progress_interval and t % args.progress_interval == 0:
                sizes = " ".join(f"{rows[b][-1][4]:.1f}" for b in np.flatnonzero(active))
                print(f"[L={L}] step {t} active={active.sum()}/{B} mean_size: {sizes}  ({time.time() - t0:.0f}s)", flush=True)
            if args.checkpoint_interval and t % args.checkpoint_interval == 0:
                for b in np.flatnonzero(active):
                    save(run_dir / f"seed_{seeds[b]}.partial.csv", rows[b])
        if t == args.steps:
            break
        states = step_b(states)
    print(f"[L={L}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
