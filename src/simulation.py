from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax import random
from jax.tree_util import register_pytree_node_class


Array = jnp.ndarray


@dataclass(frozen=True)
class SCHCParams:
    """Model parameters."""

    k: int = 1000
    n: int = 10
    L: int = 100
    mu: float = 0.002 / 0.999
    death_prob: float = 0.001


@register_pytree_node_class
@dataclass
class SCHCState:
    """Simulation state."""

    config: Array  # shape (L, L), int32; 0 means empty, 1..k are cell types
    rng: Array
    time: Array = jnp.array(0, dtype=jnp.int32)

    # Enable use as a JAX pytree (for jit/vmap/pmap).
    def tree_flatten(self):
        return (self.config, self.rng, self.time), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        config, rng, time = children
        return cls(config=config, rng=rng, time=time)


def initialize_state(key: Array, params: SCHCParams) -> SCHCState:
    """Place n random cells with random types."""

    key, choose_key, type_key = random.split(key, 3)
    config = jnp.zeros((params.L, params.L), dtype=jnp.int32)
    flat_indices = random.choice(
        choose_key, params.L * params.L, shape=(params.n,), replace=False
    )
    xs = flat_indices // params.L
    ys = flat_indices % params.L
    values = random.randint(
        type_key, (params.n,), minval=1, maxval=params.k + 1, dtype=jnp.int32
    )
    config = config.at[xs, ys].set(values)
    return SCHCState(config=config, rng=key, time=0)


def _connected_component(occupied: Array, start: Array) -> Array:
    """8-neighborhood flood fill starting from start."""

    start_mask = jnp.zeros_like(occupied, dtype=bool).at[start[0], start[1]].set(True)

    def cond(carry):
        _, frontier = carry
        return jnp.any(frontier)

    def body(carry):
        comp, frontier = carry
        neighbors = lax.reduce_window(
            frontier.astype(jnp.uint8),
            jnp.array(0, dtype=jnp.uint8),
            lax.max,
            window_dimensions=(3, 3),
            window_strides=(1, 1),
            padding="SAME",
        ).astype(bool)
        new = neighbors & occupied & ~comp
        comp = comp | new
        return comp, new

    comp, _ = lax.while_loop(cond, body, (start_mask, start_mask))
    return comp


def _bounding_box(mask: Array) -> Tuple[Array, Array]:
    L = mask.shape[0]
    grid_x = jnp.arange(L, dtype=jnp.int32)[:, None]
    grid_y = jnp.arange(L, dtype=jnp.int32)[None, :]
    valid = mask.astype(bool)
    large = jnp.array(10 * L, dtype=jnp.int32)
    mins = jnp.stack(
        [
            jnp.min(jnp.where(valid, grid_x, large)),
            jnp.min(jnp.where(valid, grid_y, large)),
        ]
    )
    maxs = jnp.stack(
        [
            jnp.max(jnp.where(valid, grid_x, -1)),
            jnp.max(jnp.where(valid, grid_y, -1)),
        ]
    )
    return mins.astype(jnp.int32), maxs.astype(jnp.int32)


def _component_fitness(config: Array, component_mask: Array) -> Array:
    """Hash-like fitness derived from relative positions and values."""

    L = config.shape[0]
    valid = component_mask.astype(bool)
    grid_x = jnp.arange(L, dtype=jnp.uint32)[:, None]
    grid_y = jnp.arange(L, dtype=jnp.uint32)[None, :]
    large = jnp.array(10 * L, dtype=jnp.uint32)
    min_x = jnp.min(jnp.where(valid, grid_x, large))
    min_y = jnp.min(jnp.where(valid, grid_y, large))
    rel_x = (grid_x - min_x).astype(jnp.uint32)
    rel_y = (grid_y - min_y).astype(jnp.uint32)
    values = config.astype(jnp.uint32)
    elements = (
        rel_x * jnp.uint32(0x9E3779B9)
        ^ rel_y * jnp.uint32(0x85EBCA6B)
        ^ values
    )
    elements = jnp.where(valid, elements, jnp.uint32(0))
    mixed = jnp.bitwise_xor.reduce(elements.reshape(-1), dtype=jnp.uint32)
    mixed = (mixed ^ jnp.uint32(0x45D9F3B)) * jnp.uint32(0x27D4EB2D)
    return mixed.astype(jnp.float32) / jnp.float32(jnp.iinfo(jnp.uint32).max)


def _mutate(values: Array, rng: Array, params: SCHCParams) -> Tuple[Array, Array]:
    rng, death_key, mutate_key, randval_key = random.split(rng, 4)
    death_roll = random.uniform(death_key, values.shape)
    mutate_roll = random.uniform(mutate_key, values.shape)
    rand_vals = random.randint(
        randval_key, values.shape, minval=1, maxval=params.k + 1, dtype=jnp.int32
    )
    after_death = jnp.where(death_roll < params.death_prob, 0, values)
    mutated = jnp.where(
        (death_roll >= params.death_prob) & (mutate_roll < params.mu),
        rand_vals,
        after_death,
    )
    return mutated, rng


def _rewrite(
    config: Array, r1: Tuple[Array, Array], r2: Tuple[Array, Array], rng: Array, params: SCHCParams
) -> Tuple[Array, Array]:
    """Copy r1 pattern onto the area centered on r2 with mutation."""

    L = config.shape[0]
    r1_min, r1_max = r1
    r2_min, r2_max = r2
    r1_center = jnp.rint((r1_min + r1_max) / 2.0).astype(jnp.int32)
    r2_center = jnp.rint((r2_min + r2_max) / 2.0).astype(jnp.int32)
    start = r2_center + r1_min - r1_center - jnp.array([1, 1], dtype=jnp.int32)
    block_min = r1_min - 1
    block_max = r1_max + 1
    block_shape = block_max - block_min + 1

    max_range = params.L + 2  # upper bound for possible block extents
    x_offsets = jnp.arange(max_range, dtype=jnp.int32)
    y_offsets = jnp.arange(max_range, dtype=jnp.int32)
    gx, gy = jnp.meshgrid(block_min[0] + x_offsets, block_min[1] + y_offsets, indexing="ij")
    tx, ty = jnp.meshgrid(start[0] + x_offsets, start[1] + y_offsets, indexing="ij")

    within_shape = (x_offsets[:, None] < block_shape[0]) & (y_offsets[None, :] < block_shape[1])
    valid = (
        within_shape
        & (gx >= 0)
        & (gx < L)
        & (gy >= 0)
        & (gy < L)
        & (tx >= 0)
        & (tx < L)
        & (ty >= 0)
        & (ty < L)
    )

    source_vals = jnp.where(valid, config[jnp.clip(gx, 0, L - 1), jnp.clip(gy, 0, L - 1)], 0).astype(
        jnp.int32
    )
    mutated, rng = _mutate(source_vals.reshape(-1), rng, params)
    mutated = mutated.reshape(source_vals.shape)

    tx_idx = jnp.where(valid, jnp.clip(tx, 0, L - 1), 0)
    ty_idx = jnp.where(valid, jnp.clip(ty, 0, L - 1), 0)
    updates = jnp.where(valid, mutated, config[tx_idx, ty_idx])
    new_config = config.at[tx_idx, ty_idx].set(updates)
    return new_config, rng


def _update_once(state: SCHCState, params: SCHCParams) -> Tuple[SCHCState, Array]:
    """Single contest/update; structured to be JIT-friendly."""

    config, rng = state.config, state.rng
    occupied = config > 0
    num_active = occupied.sum().astype(jnp.int32)

    def no_active_branch(_):
        return SCHCState(config=config, rng=rng, time=state.time), jnp.float32(1.0)

    def active_branch(_):
        rng1, pick_key1, pick_key2 = random.split(rng, 3)
        active_indices = jnp.argwhere(
            occupied, size=params.L * params.L, fill_value=jnp.array(0, dtype=jnp.int32)
        ).astype(jnp.int32)
        active_len = num_active
        idx1 = random.randint(pick_key1, (), 0, active_len)
        idx2 = random.randint(pick_key2, (), 0, active_len)
        start1 = active_indices[idx1]
        start2 = active_indices[idx2]

        comp1 = _connected_component(occupied, start1)
        comp2 = _connected_component(occupied, start2)
        r1 = _bounding_box(comp1)
        r2 = _bounding_box(comp2)

        f1 = _component_fitness(config, comp1)
        f2 = _component_fitness(config, comp2)
        fitnesses = jnp.stack([f1, f2])
        max_fit = jnp.max(fitnesses)
        winners = jnp.where(fitnesses == max_fit, size=2, fill_value=0)[0]
        rng2, win_key = random.split(rng1)
        win_idx = random.randint(win_key, (), 0, winners.shape[0])
        winner_choice = winners[win_idx]

        comp_sizes = jnp.stack([comp1.sum(), comp2.sum()]).astype(jnp.float32)
        winner_prop = comp_sizes[winner_choice] / jnp.maximum(num_active.astype(jnp.float32), 1.0)

        new_config, rng_out = lax.cond(
            winner_choice == 0,
            lambda _: _rewrite(config, r1, r2, rng2, params),
            lambda _: _rewrite(config, r2, r1, rng2, params),
            operand=None,
        )
        return SCHCState(config=new_config, rng=rng_out, time=state.time), winner_prop

    return lax.cond(num_active == 0, no_active_branch, active_branch, operand=None)


def advance_one_step(state: SCHCState, params: SCHCParams) -> SCHCState:
    """Equivalent to Mathematica's onestep: advance until elapsed >= 1, device-side loop."""

    def cond_fn(carry):
        _, elapsed = carry
        return elapsed < jnp.float32(1.0)

    def body_fn(carry):
        working_state, elapsed = carry
        working_state, winner_prop = _update_once(working_state, params)
        winner_prop = jnp.maximum(winner_prop, jnp.float32(1e-6))
        return working_state, elapsed + winner_prop

    init = (state, jnp.float32(0.0))
    working_state, _ = lax.while_loop(cond_fn, body_fn, init)
    return SCHCState(config=working_state.config, rng=working_state.rng, time=working_state.time + 1)


def run_steps(
    state: SCHCState,
    params: SCHCParams,
    steps: int,
    callback: Optional[Callable[[int, SCHCState], None]] = None,
) -> SCHCState:
    """Run multiple onestep iterations."""

    working_state = state
    for ii in range(steps):
        working_state = advance_one_step(working_state, params)
        if callback is not None:
            callback(ii, working_state)
    return working_state


# JIT-compiled helpers for device側ループをまとめる
advance_one_step_jit = jax.jit(advance_one_step, static_argnums=1)


def _run_steps_fori(state: SCHCState, params: SCHCParams, steps: int) -> SCHCState:
    """Run multiple onestep iterations using lax.fori_loop (callbacks unsupported)."""

    def body_fn(ii, carry):
        _ = ii
        return advance_one_step(carry, params)

    return lax.fori_loop(0, steps, body_fn, state)


run_steps_jit = jax.jit(_run_steps_fori, static_argnums=(1, 2))
