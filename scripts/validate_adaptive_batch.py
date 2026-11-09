#!/usr/bin/env python3
"""Check state, PRNG and event records across adaptive vmap splits."""
from __future__ import annotations

import sys
import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.adaptive_batch import run_batch
from scripts import compare_original_schc as comparison
from src.simulation import SCHCParams, advance_one_step, initialize_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = []
    reference = getattr(comparison, "run_batch_reference", comparison.run_batch)
    for settings in ({}, {"D": 2, "P": .6}, {"D": 2, "P": .6, "dyadic_rule": "as_coded"},
                     {"winner_rule": "legacy", "score": "xor"}):
        params = SCHCParams(L=8, **settings)
        seeds, steps = list(range(6)), 25
        expected = reference(params, seeds, steps)
        state = jax.jit(jax.vmap(lambda key: initialize_state(key, params)))(
            jnp.stack([random.PRNGKey(s) for s in seeds]))
        advance = jax.jit(jax.vmap(lambda state: advance_one_step(state, params)))
        configs, keys = [], []
        for _ in range(steps):
            state = advance(state)
            configs.append(np.asarray(state.config))
            keys.append(np.asarray(state.rng))
        sizes = []
        def check(t, indices, state):
            sizes.append(len(indices))
            np.testing.assert_array_equal(state.config, configs[t][indices])
            np.testing.assert_array_equal(state.rng, keys[t][indices])
        got = run_batch(params, seeds, steps, split_threshold=3, state_callback=check)
        np.testing.assert_array_equal(np.array(got), np.array(expected))
        assert min(sizes) == 1, "test must exercise singleton splits"
        cases.append(dict(settings=settings, seeds=seeds, steps=steps, exact_state_prng_events=True))
        print(f"Exact state/PRNG/event agreement: {settings or 'original/mix global'}", flush=True)
    print(f"All four adaptive-batch cases passed on {jax.devices()}", flush=True)
    if args.out:
        sources = [ROOT / "src/simulation.py", ROOT / "scripts/adaptive_batch.py",
                   ROOT / "scripts/compare_original_schc.py", Path(__file__)]
        report = dict(jax_version=jax.__version__, devices=list(map(str, jax.devices())),
                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), cases=cases,
                      source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sources})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
