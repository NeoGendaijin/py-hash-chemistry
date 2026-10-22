"""Device-side component metrics for SCHC configurations (open boundaries).

label_components: 8-connected component labels by min-label propagation with
pointer jumping (each label is the flat index + 1 of the component's smallest cell).
component_metrics: per-configuration summary used by the transition scans
(sizes, "mix" scores, cell types present, compacted pattern ids).
"""

from typing import Dict

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .simulation import _fmix32

Array = jnp.ndarray


def label_components(occupied: Array) -> Array:
    L = occupied.shape[0]
    inf = np.int32(L * L + 1)  # concrete scalar: reduce_window needs a static init value
    idx = jnp.arange(1, L * L + 1, dtype=jnp.int32).reshape(L, L)
    lab = jnp.where(occupied, idx, inf)

    def body(carry):
        lab, _ = carry
        m = lax.reduce_window(lab, inf, lax.min, (3, 3), (1, 1), "SAME")
        new = jnp.where(occupied, jnp.minimum(lab, m), inf)
        flat = new.reshape(-1)
        new = jnp.where(occupied, flat[jnp.clip(new - 1, 0, L * L - 1)].reshape(L, L), inf)
        return new, jnp.any(new != lab)

    lab, _ = lax.while_loop(lambda c: c[1], body, (lab, jnp.array(True)))
    return lab


def _segment_mix_hash(values, rel_x, rel_y, seg, num, seed: int):
    """Per-segment version of simulation._component_hash(..., score="mix", seed)."""
    h = _fmix32(values + jnp.uint32((0x9E3779B9 + 0x632BE5AB * seed) & 0xFFFFFFFF))
    h = _fmix32(h ^ (rel_x * jnp.uint32(0x85EBCA6B) + jnp.uint32(0x7F4A7C15)))
    h = _fmix32(h ^ (rel_y * jnp.uint32(0xC2B2AE35) + jnp.uint32(0x165667B1)))
    total = jax.ops.segment_sum(h, seg, num_segments=num)
    return _fmix32(total ^ jnp.uint32((0x27D4EB2F * (seed + 1)) & 0xFFFFFFFF))


def component_metrics(config: Array, k: int, max_components: int) -> Dict[str, Array]:
    """Summary of one configuration.

    Returns n_active, n_components, mean_size, max_size, mean/max "mix" score over
    components, types (bool (k+1,), cell types present), and pattern ids
    (uint32 (max_components, 2), first n_components rows valid).
    """
    L = config.shape[0]
    occupied = config > 0
    lab = label_components(occupied)
    num = L * L + 2
    seg = lab.reshape(-1)
    sizes = jax.ops.segment_sum(occupied.reshape(-1).astype(jnp.int32), seg, num_segments=num)
    is_root = (jnp.arange(num) >= 1) & (jnp.arange(num) <= L * L) & (sizes > 0)
    n_comp = is_root.sum(dtype=jnp.int32)
    n_active = occupied.sum(dtype=jnp.int32)

    gx = jnp.broadcast_to(jnp.arange(L, dtype=jnp.int32)[:, None], (L, L)).reshape(-1)
    gy = jnp.broadcast_to(jnp.arange(L, dtype=jnp.int32)[None, :], (L, L)).reshape(-1)
    min_x = jax.ops.segment_min(gx, seg, num_segments=num)
    min_y = jax.ops.segment_min(gy, seg, num_segments=num)
    rel_x = (gx - min_x[seg]).astype(jnp.uint32)
    rel_y = (gy - min_y[seg]).astype(jnp.uint32)
    values = config.reshape(-1).astype(jnp.uint32)

    score_h = _segment_mix_hash(values, rel_x, rel_y, seg, num, seed=0)
    score = score_h.astype(jnp.float32) / jnp.float32(jnp.iinfo(jnp.uint32).max)
    id_a = _segment_mix_hash(values, rel_x, rel_y, seg, num, seed=101)
    id_b = _segment_mix_hash(values, rel_x, rel_y, seg, num, seed=202)

    roots = jnp.nonzero(is_root, size=max_components, fill_value=0)[0]
    denom = jnp.maximum(n_comp, 1).astype(jnp.float32)
    return {
        "n_active": n_active,
        "n_components": n_comp,
        "mean_size": n_active.astype(jnp.float32) / denom,
        "max_size": jnp.max(jnp.where(is_root, sizes, 0)),
        "mean_fitness": jnp.where(is_root, score, 0.0).sum() / denom,
        "max_fitness": jnp.max(jnp.where(is_root, score, 0.0)),
        "types": jnp.zeros(k + 1, dtype=bool).at[config.reshape(-1)].set(True),
        "pattern_ids": jnp.stack([id_a[roots], id_b[roots]], axis=1),
    }
