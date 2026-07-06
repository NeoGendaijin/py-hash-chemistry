#!/usr/bin/env python3
"""Regenerate grid-configuration snapshots for the SCHC hero figures.

Runs the exact SCHC dynamics (default parameters) for the representative
compact run (L=200, seed 0) and runaway run (L=400, seed 8), saving the raw
config arrays at several timepoints so the figure script can pick a clean,
consistently-timed sequence. Only the cheap competition/replication is run
(no per-step component metrics), so reaching t=20,000 is fast.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from jax import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import SCHCParams, advance_one_step_jit, initialize_state

SAVE_STEPS = [2000, 6000, 10000, 20000]
OUT = ROOT / "results" / "snapshots_hero"


def run(L, seed):
    params = SCHCParams(L=L)  # defaults: k=1000, n=10, mu=0.002/0.999, death=0.001
    state = initialize_state(random.PRNGKey(seed), params)
    outdir = OUT / f"L{L}_seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    save_set = set(SAVE_STEPS)
    for t in range(max(SAVE_STEPS) + 1):
        if t in save_set:
            np.save(outdir / f"step_{t:05d}.npy", np.array(state.config))
            print(f"[L={L} seed={seed}] saved t={t}", flush=True)
        if t == max(SAVE_STEPS):
            break
        state = advance_one_step_jit(state, params)


if __name__ == "__main__":
    run(200, 0)
    run(400, 8)
    print("[done] snapshots_hero regenerated", flush=True)
