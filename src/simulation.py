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
    periodic: bool = False
    # Winner selection.
    #   "legacy":   the rule behind all JAX results published so far. When the second
    #               contestant scores strictly higher it wins only half of the time, so
    #               the higher-scoring component wins 3/4 of non-tied contests.
    #   "original": the Mathematica SCHC rule, RandomChoice[Position[#, Max[#]]]:
    #               the higher score always wins, exact ties are broken uniformly.
    winner_rule: str = "original"
    # Score function.
    #   "xor": the published surrogate. It XOR-reduces per-cell terms, so the score depends
    #          only on the XOR of all cell types and on row/column parities of the shape.
    #   "mix": sum of strongly mixed per-cell hashes of (rel_x, rel_y, type), so it depends
    #          on which type sits where (closer to Mathematica's Hash of the sorted list).
    score: str = "mix"
    # Reproduce the scatter used for the archived JAX runs, which can drop a
    # valid update to (0, 0). New simulations drop invalid indices instead.
    legacy_corner: bool = False
    # Spatial-dyadic extension (Sayama, ALIFE 2026 LBA).
    # D: the second cell is drawn uniformly from the active cells within Chebyshev
    #    distance D of the first (inclusive, the first cell itself included, window
    #    clipped at open edges / wrapped when periodic). None = global (original SCHC).
    D: Optional[int] = None
    # P: probability that a contest is dyadic: the ordered pair of patterns is hashed
    #    and the first pattern wins if that hash exceeds 0.5.
    P: float = 0.0
    #   "intended": pair order is canonical (independent of sampling order), so a given
    #               pair always has the same winner, as described in the papers;
    #   "as_coded": the first *sampled* contestant wins if the pair hash exceeds 0.5,
    #               as in the Mathematica code, where the order swap never runs
    #               (`reserved = True` typo for `reversed = True`).
    dyadic_rule: str = "intended"

    def __post_init__(self):
        if self.periodic:
            raise NotImplementedError(
                "Periodic scoring/copying is not validated; the revised paper uses open boundaries."
            )
        if self.winner_rule not in ("legacy", "original"):
            raise ValueError(f"unknown winner_rule: {self.winner_rule!r}")
        if self.score not in ("xor", "mix"):
            raise ValueError(f"unknown score: {self.score!r}")
        if self.D is not None and (not isinstance(self.D, int) or self.D < 1):
            raise ValueError(f"D must be a positive int or None, got {self.D!r}")
        if not 0.0 <= self.P <= 1.0:
            raise ValueError(f"P must be in [0, 1], got {self.P!r}")
        if self.dyadic_rule not in ("intended", "as_coded"):
            raise ValueError(f"unknown dyadic_rule: {self.dyadic_rule!r}")

    @property
    def global_competition(self) -> bool:
        """True when the D-window covers the whole grid, i.e. the original global pick."""
        if self.D is None:
            return True
        return self.D >= (self.L // 2 if self.periodic else self.L - 1)


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


def _connected_component(occupied: Array, start: Array, periodic: bool = False) -> Array:
    """8-neighborhood flood fill starting from start.

    periodic=False: open boundaries (reduce_window with zero padding, no wrap).
    periodic=True : toroidal neighborhood via wrapping rolls, so a component
                    spanning the grid seam is recognized as a single component.
    """

    start_mask = jnp.zeros_like(occupied, dtype=bool).at[start[0], start[1]].set(True)

    def cond(carry):
        _, frontier = carry
        return jnp.any(frontier)

    def body(carry):
        comp, frontier = carry
        if periodic:
            neighbors = jnp.zeros_like(frontier)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbors = neighbors | jnp.roll(jnp.roll(frontier, dx, axis=0), dy, axis=1)
        else:
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


def _fmix32(h: Array) -> Array:
    """MurmurHash3 32-bit finalizer (bijective avalanche mixer on uint32)."""

    h = h ^ (h >> 16)
    h = h * jnp.uint32(0x85EBCA6B)
    h = h ^ (h >> 13)
    h = h * jnp.uint32(0xC2B2AE35)
    return h ^ (h >> 16)


def _component_hash(config: Array, component_mask: Array, score: str = "xor", seed: int = 0) -> Array:
    """uint32 hash of a component from its translation-normalized cells and types."""

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
    if score == "xor":
        elements = (
            rel_x * jnp.uint32(0x9E3779B9)
            ^ rel_y * jnp.uint32(0x85EBCA6B)
            ^ values
        )
        elements = jnp.where(valid, elements, jnp.uint32(0))
        mixed = jnp.bitwise_xor.reduce(elements.reshape(-1), dtype=jnp.uint32)
        return (mixed ^ jnp.uint32(0x45D9F3B)) * jnp.uint32(0x27D4EB2D)
    # "mix": nonlinear per-cell hash, combined by an order-free wrapping sum.
    h = _fmix32(values + jnp.uint32((0x9E3779B9 + 0x632BE5AB * seed) & 0xFFFFFFFF))
    h = _fmix32(h ^ (rel_x * jnp.uint32(0x85EBCA6B) + jnp.uint32(0x7F4A7C15)))
    h = _fmix32(h ^ (rel_y * jnp.uint32(0xC2B2AE35) + jnp.uint32(0x165667B1)))
    h = jnp.where(valid, h, jnp.uint32(0))
    total = jnp.sum(h, dtype=jnp.uint32)
    return _fmix32(total ^ jnp.uint32((0x27D4EB2F * (seed + 1)) & 0xFFFFFFFF))


def _hash_to_fitness(h: Array) -> Array:
    return h.astype(jnp.float32) / jnp.float32(jnp.iinfo(jnp.uint32).max)


def _component_fitness(config: Array, component_mask: Array, score: str = "xor") -> Array:
    """Hash-like fitness in [0, 1] derived from relative positions and values."""

    return _hash_to_fitness(_component_hash(config, component_mask, score))


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

    if params.periodic:
        # Toroidal boundaries: wrap source and target coordinates modulo L,
        # so a structure's footprint is never clipped at the grid edge.
        gx_idx = jnp.mod(gx, L)
        gy_idx = jnp.mod(gy, L)
        tx_idx = jnp.mod(tx, L)
        ty_idx = jnp.mod(ty, L)
        valid = within_shape
        source_vals = jnp.where(valid, config[gx_idx, gy_idx], 0).astype(jnp.int32)
        mutated, rng = _mutate(source_vals.reshape(-1), rng, params)
        mutated = mutated.reshape(source_vals.shape)
        tx_idx = jnp.where(valid, tx_idx, 0)
        ty_idx = jnp.where(valid, ty_idx, 0)
        updates = jnp.where(valid, mutated, config[tx_idx, ty_idx])
        new_config = config.at[tx_idx, ty_idx].set(updates)
        return new_config, rng

    # Open boundaries: any cell falling outside the L x L domain is clipped (dropped).
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

    if params.legacy_corner:
        tx_idx = jnp.where(valid, jnp.clip(tx, 0, L - 1), 0)
        ty_idx = jnp.where(valid, jnp.clip(ty, 0, L - 1), 0)
        updates = jnp.where(valid, mutated, config[tx_idx, ty_idx])
        new_config = config.at[tx_idx, ty_idx].set(updates)
    else:
        # Padded entries must not compete with a real update to the corner.
        tx_idx = jnp.where(valid, tx, L)
        ty_idx = jnp.where(valid, ty, L)
        new_config = config.at[tx_idx, ty_idx].set(mutated, mode="drop")
    return new_config, rng


def _pick_local(occupied: Array, start1: Array, key: Array, params: SCHCParams) -> Array:
    """Uniform active cell within Chebyshev distance D of start1 (start1 itself included).

    Mathematica: RandomChoice[Nearest[activeCells, cell1, {All, D}, DistanceFunction -> ChessboardDistance]].
    Draws randint(key, 0, n_window) and takes that cell in row-major order, so a window
    covering every active cell reproduces the global pick exactly.
    """

    L = params.L
    dx = jnp.abs(jnp.arange(L, dtype=jnp.int32)[:, None] - start1[0])
    dy = jnp.abs(jnp.arange(L, dtype=jnp.int32)[None, :] - start1[1])
    if params.periodic:
        dx = jnp.minimum(dx, L - dx)
        dy = jnp.minimum(dy, L - dy)
    window = occupied & (dx <= params.D) & (dy <= params.D)
    r = random.randint(key, (), 0, window.sum(dtype=jnp.int32))
    j = jnp.argmax(jnp.cumsum(window.reshape(-1), dtype=jnp.int32) > r).astype(jnp.int32)
    return jnp.stack([j // L, j % L])


def _dyadic_winner(config: Array, comp1: Array, comp2: Array, rule: str) -> Array:
    """Winner index (0 or 1) of a dyadic contest.

    The pair of patterns is put in a canonical order (by a 64-bit pattern id), the
    ordered pair is hashed, and the first pattern of that order wins if the hash
    exceeds 0.5 ("intended"). With rule "as_coded" the index refers to the sampling
    order instead, as in the Mathematica code where the order swap never runs.
    """

    a1 = _component_hash(config, comp1, "mix", seed=101)
    b1 = _component_hash(config, comp1, "mix", seed=202)
    a2 = _component_hash(config, comp2, "mix", seed=101)
    b2 = _component_hash(config, comp2, "mix", seed=202)
    one_first = (a1 < a2) | ((a1 == a2) & (b1 <= b2))
    fa, fb = jnp.where(one_first, a1, a2), jnp.where(one_first, b1, b2)
    sa, sb = jnp.where(one_first, a2, a1), jnp.where(one_first, b2, b1)
    h = _fmix32(fa + jnp.uint32(0x3C6EF372))
    h = _fmix32(h ^ fb)
    h = _fmix32(h + sa * jnp.uint32(0x9E3779B9))
    h = _fmix32(h ^ (sb + jnp.uint32(0xA54FF53A)))
    first_wins = _hash_to_fitness(h) > 0.5
    if rule == "as_coded":
        return jnp.where(first_wins, 0, 1)
    return jnp.where(first_wins == one_first, 0, 1)


def _contest(state: SCHCState, params: SCHCParams) -> Tuple[SCHCState, Array, Tuple[Array, Array, Array]]:
    """Single contest/update; structured to be JIT-friendly.

    Returns (new_state, winner_prop, (winner_hash, winner_size, winner_mask)); the last
    three describe the winning component in the pre-contest configuration.
    """

    config, rng = state.config, state.rng
    occupied = config > 0
    num_active = occupied.sum().astype(jnp.int32)

    def no_active_branch(_):
        no_winner = (jnp.uint32(0), jnp.float32(0.0), jnp.zeros_like(occupied))
        return SCHCState(config=config, rng=rng, time=state.time), jnp.float32(1.0), no_winner

    def active_branch(_):
        rng1, pick_key1, pick_key2 = random.split(rng, 3)
        active_indices = jnp.argwhere(
            occupied, size=params.L * params.L, fill_value=jnp.array(0, dtype=jnp.int32)
        ).astype(jnp.int32)
        active_len = num_active
        idx1 = random.randint(pick_key1, (), 0, active_len)
        start1 = active_indices[idx1]
        if params.global_competition:
            idx2 = random.randint(pick_key2, (), 0, active_len)
            start2 = active_indices[idx2]
        else:
            start2 = _pick_local(occupied, start1, pick_key2, params)

        comp1 = _connected_component(occupied, start1, params.periodic)
        comp2 = _connected_component(occupied, start2, params.periodic)
        r1 = _bounding_box(comp1)
        r2 = _bounding_box(comp2)

        h1 = _component_hash(config, comp1, params.score)
        h2 = _component_hash(config, comp2, params.score)
        f1 = _hash_to_fitness(h1)
        f2 = _hash_to_fitness(h2)
        rng2, win_key = random.split(rng1)
        if params.winner_rule == "legacy":
            fitnesses = jnp.stack([f1, f2])
            max_fit = jnp.max(fitnesses)
            # size=2 pads with index 0, so for f2 > f1 this is [1, 0] and comp1 still wins half the time.
            winners = jnp.where(fitnesses == max_fit, size=2, fill_value=0)[0]
            win_idx = random.randint(win_key, (), 0, winners.shape[0])
            winner_choice = winners[win_idx]
        else:
            tie_break = random.randint(win_key, (), 0, 2)
            winner_choice = jnp.where(f1 > f2, 0, jnp.where(f2 > f1, 1, tie_break))
        if params.P > 0:
            dyadic = random.uniform(random.fold_in(win_key, 7)) < params.P
            winner_choice = jnp.where(
                dyadic, _dyadic_winner(config, comp1, comp2, params.dyadic_rule), winner_choice
            )

        comp_sizes = jnp.stack([comp1.sum(), comp2.sum()]).astype(jnp.float32)
        winner_prop = comp_sizes[winner_choice] / jnp.maximum(num_active.astype(jnp.float32), 1.0)

        new_config, rng_out = lax.cond(
            winner_choice == 0,
            lambda _: _rewrite(config, r1, r2, rng2, params),
            lambda _: _rewrite(config, r2, r1, rng2, params),
            operand=None,
        )
        winner = (
            jnp.where(winner_choice == 0, h1, h2),
            comp_sizes[winner_choice],
            jnp.where(winner_choice == 0, comp1, comp2),
        )
        return SCHCState(config=new_config, rng=rng_out, time=state.time), winner_prop, winner

    return lax.cond(num_active == 0, no_active_branch, active_branch, operand=None)


def _update_once(state: SCHCState, params: SCHCParams) -> Tuple[SCHCState, Array]:
    """Single contest/update; structured to be JIT-friendly."""

    new_state, winner_prop, _ = _contest(state, params)
    return new_state, winner_prop


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


def _num_digits(x: Array) -> Array:
    """Decimal digits of non-negative int32 values."""

    return 1 + sum((x >= 10**j).astype(jnp.int32) for j in range(1, 10))


def advance_one_step_logged(
    state: SCHCState, params: SCHCParams, seen_types: Array, max_events: int
) -> Tuple[SCHCState, Array, dict]:
    """advance_one_step that also aggregates the contest winners of the step.

    The dynamics are identical to advance_one_step. The statistics follow the
    per-event log of the Mathematica code (Write[file, {time, winnerPattern,
    winnerFitness}]) as aggregated by SCHC-systematic-sim-data-analysis.nb.
    seen_types (bool, shape (k + 1,)) accumulates the cell types of winner
    patterns across steps. Per-event winner-pattern IDs (two independent uint32
    hashes, for counting distinct pattern types on the host) are written to a
    buffer of max_events rows; n_events > max_events means it overflowed.
    """

    L = params.L
    uint_max = jnp.float32(jnp.iinfo(jnp.uint32).max)
    grid_x = jnp.arange(L, dtype=jnp.int32)[:, None]
    grid_y = jnp.arange(L, dtype=jnp.int32)[None, :]
    time_digits = _num_digits(state.time)
    n_active = (state.config > 0).sum().astype(jnp.int32)

    def cond_fn(carry):
        return carry["elapsed"] < jnp.float32(1.0)

    def body_fn(carry):
        working_state = carry["state"]
        old_config = working_state.config
        working_state, winner_prop, (w_hash, w_size, w_mask) = _contest(working_state, params)
        winner_prop = jnp.maximum(winner_prop, jnp.float32(1e-6))
        real = w_size > 0  # False only for the no-active-cell branch

        # Winner pattern in relative coordinates, as written to the Mathematica log.
        min_x = jnp.min(jnp.where(w_mask, grid_x, 10 * L))
        min_y = jnp.min(jnp.where(w_mask, grid_y, 10 * L))
        cell_chars = 8 + _num_digits(grid_x - min_x) + _num_digits(grid_y - min_y) + _num_digits(old_config)
        size_i = w_size.astype(jnp.int32)
        # "{t, {{{dx, dy}, v}, ...}, 0.dddddddd}\n"
        event_bytes = time_digits + 19 + jnp.where(w_mask, cell_chars, 0).sum() + 2 * (size_i - 1)
        pattern_id = jnp.stack([
            _component_hash(old_config, w_mask, "mix", seed=101),
            _component_hash(old_config, w_mask, "mix", seed=202),
        ])
        slot = jnp.minimum(carry["n_events"], max_events - 1)
        return {
            "state": working_state,
            "elapsed": carry["elapsed"] + winner_prop,
            "n_events": carry["n_events"] + real.astype(jnp.int32),
            "max_hash": jnp.where(real, jnp.maximum(carry["max_hash"], w_hash), carry["max_hash"]),
            # 1 - fitness, kept separately so that values near 1 keep their precision
            "sum_gap": carry["sum_gap"] + jnp.where(real, (jnp.uint32(0xFFFFFFFF) - w_hash).astype(jnp.float32) / uint_max, 0.0),
            "sum_size": carry["sum_size"] + size_i,
            "max_size": jnp.maximum(carry["max_size"], size_i),
            "log_bytes": carry["log_bytes"] + jnp.where(real, event_bytes, 0),
            "seen_types": carry["seen_types"].at[jnp.where(w_mask, old_config, 0)].set(True),
            "ids": jnp.where(real, carry["ids"].at[slot].set(pattern_id), carry["ids"]),
        }

    init = {
        "state": state,
        "elapsed": jnp.float32(0.0),
        "n_events": jnp.int32(0),
        "max_hash": jnp.uint32(0),
        "sum_gap": jnp.float32(0.0),
        "sum_size": jnp.int32(0),
        "max_size": jnp.int32(0),
        "log_bytes": jnp.int32(0),
        "seen_types": seen_types,
        "ids": jnp.zeros((max_events, 2), dtype=jnp.uint32),
    }
    out = lax.while_loop(cond_fn, body_fn, init)
    new_state = SCHCState(config=out["state"].config, rng=out["state"].rng, time=out["state"].time + 1)
    stats = {k: out[k] for k in ("n_events", "max_hash", "sum_gap", "sum_size", "max_size", "log_bytes", "ids")}
    stats["n_active"] = n_active
    return new_state, out["seen_types"], stats


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
