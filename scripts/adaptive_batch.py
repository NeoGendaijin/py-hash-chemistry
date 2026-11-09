"""Split large runs out of vmap batches without changing any per-run operation."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from src.simulation import advance_one_step_logged, initialize_state


def run_batch(params, seeds, steps, split_threshold=1000, state_callback=None):
    max_events = params.L * params.L + 1
    init = jax.jit(jax.vmap(lambda key: initialize_state(key, params)))
    step = jax.jit(jax.vmap(lambda state, seen: advance_one_step_logged(state, params, seen, max_events)))
    states = init(jnp.stack([random.PRNGKey(s) for s in seeds]))
    seen = jnp.zeros((len(seeds), params.k + 1), dtype=bool)
    groups = [(np.arange(len(seeds)), states, seen)]
    patterns, rows = [set() for _ in seeds], [[] for _ in seeds]
    log_bytes = np.zeros(len(seeds), dtype=np.int64)
    uint_max = float(0xFFFFFFFF)
    for t in range(steps):
        next_groups = []
        for indices, states, seen in groups:
            states, seen, st = step(states, seen)
            st = jax.device_get(st)
            n_types = np.asarray(seen[:, 1:].sum(axis=1))
            if state_callback is not None:
                state_callback(t, indices, states)
            for b, index in enumerate(indices):
                n = int(st["n_events"][b])
                if n > max_events:
                    raise RuntimeError(f"event buffer overflow: seed {seeds[index]} step {t} n={n}")
                ids = st["ids"][b, :n].astype(np.uint64)
                patterns[index].update(((ids[:, 0] << np.uint64(32)) | ids[:, 1]).tolist())
                log_bytes[index] += int(st["log_bytes"][b])
                if n:
                    max_gap = (uint_max - float(st["max_hash"][b])) / uint_max
                    mean_gap = float(st["sum_gap"][b]) / n
                    vals = (1 - max_gap, 1 - mean_gap, max_gap, mean_gap, int(st["sum_size"][b]),
                            int(st["max_size"][b]), int(st["sum_size"][b]) / n)
                else:
                    vals = (np.nan,) * 4 + (0, 0, np.nan)
                rows[index].append((t, *vals, int(n_types[b]), len(patterns[index]), n,
                                    int(st["n_active"][b]), int(log_bytes[index])))
            large = np.flatnonzero(np.asarray(st["max_size"]) >= split_threshold)
            if len(indices) > 1 and len(large):
                small = np.flatnonzero(np.asarray(st["max_size"]) < split_threshold)
                partitions = ([small] if len(small) else []) + [np.array([b]) for b in large]
                for selection in partitions:
                    # Keep the complete state and PRNG; splitting never restarts a run.
                    group_state = jax.tree.map(lambda a: a[selection], states)
                    next_groups.append((indices[selection], group_state, seen[selection]))
            else:
                next_groups.append((indices, states, seen))
        groups = next_groups
        if (t + 1) % 100 == 0:
            print(f"step {t + 1}/{steps}; batch sizes {[len(g[0]) for g in groups]}", flush=True)
    return rows
