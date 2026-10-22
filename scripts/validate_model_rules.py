#!/usr/bin/env python3
"""Small, deterministic rule checks, including corners and C/JAX agreement.

Run with JAX_PLATFORMS=cpu python scripts/validate_model_rules.py.
This validates implementation rules, not statistical equivalence of ensembles.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import fast
from src.simulation import (SCHCParams, _component_hash,
                            _connected_component, _dyadic_winner, _rewrite)


def numpy_contest(grid, start1, start2, coin, rule, score):
    labels, _ = ndimage.label(grid > 0, structure=np.ones((3, 3)))
    masks = [labels == labels.flat[s] for s in (start1, start2)]
    cells = [np.flatnonzero(m).astype(np.int32) for m in masks]
    hashes = [fast.component_hash(grid, c, score=score) for c in cells]
    f1, f2 = [np.float32(h) / np.float32(2**32) for h in hashes]
    winner = (0 if f1 > f2 else (1 if f2 > f1 else coin)) if rule == "original" else (0 if f1 > f2 else coin)
    points = [np.argwhere(m) for m in masks]
    boxes = [(p.min(0), p.max(0)) for p in points]
    wm, wx = boxes[winner]
    lm, lx = boxes[1 - winner]
    start = np.rint((lm + lx) / 2).astype(int) + wm - np.rint((wm + wx) / 2).astype(int) - 1
    expected = grid.copy()
    for ox in range(int(wx[0] - wm[0] + 3)):
        for oy in range(int(wx[1] - wm[1] + 3)):
            gx, gy = wm - 1 + [ox, oy]
            tx, ty = start + [ox, oy]
            if all(0 <= a < len(grid) for a in (gx, gy, tx, ty)):
                expected[tx, ty] = grid[gx, gy]
    return expected, boxes[winner], boxes[1 - winner]


class ModelRules(unittest.TestCase):
    def test_defaults_and_unsupported_boundaries(self):
        p = SCHCParams()
        self.assertEqual((p.winner_rule, p.score, p.legacy_corner), ("original", "mix", False))
        with self.assertRaises(NotImplementedError):
            SCHCParams(periodic=True)

    def test_hash_matches_c_and_is_translation_invariant(self):
        rng = np.random.default_rng(20261023)
        for _ in range(40):
            grid = np.zeros((12, 12), np.int32)
            grid[2:6, 2:6] = rng.integers(1, 1001, (4, 4))
            mask = grid > 0
            cells = np.flatnonzero(mask).astype(np.int32)
            for score in ("xor", "mix"):
                for seed in (0, 101, 202):
                    jh = int(_component_hash(jnp.array(grid), jnp.array(mask), score, seed))
                    self.assertEqual(jh, fast.component_hash(grid, cells, score, seed))
                    shifted = np.roll(grid, (3, 4), axis=(0, 1))
                    self.assertEqual(jh, int(_component_hash(jnp.array(shifted), jnp.array(shifted > 0), score, seed)))

    def test_open_components_include_diagonals(self):
        grid = np.eye(8, dtype=bool)
        grid[0, 7] = True
        got = np.asarray(_connected_component(jnp.array(grid), jnp.array([0, 0])))
        self.assertEqual(int(got.sum()), 8)
        self.assertFalse(got[0, 7])

    def test_corner_safe_rewrite_matches_independent_reference(self):
        rng = np.random.default_rng(23)
        checks = 0
        for _ in range(12):
            grid = np.zeros((10, 10), np.int32)
            grid[0, 0] = int(rng.integers(1, 1001))
            grid[4:7, 4:7] = rng.integers(1, 1001, (3, 3))
            grid[9, 9] = int(rng.integers(1, 1001))
            for a, b in ((0, 44), (44, 0), (99, 44), (44, 99), (0, 0), (44, 44)):
                for coin in (0, 1):
                    for rule in ("original", "legacy"):
                        for score in ("mix", "xor"):
                            expected, wb, lb = numpy_contest(grid, a, b, coin, rule, score)
                            cpu = fast.forced_contest(grid, a, b, coin, winner_rule=rule, score=score)
                            np.testing.assert_array_equal(cpu, expected)
                            params = SCHCParams(L=10, mu=0, death_prob=0, winner_rule=rule, score=score)
                            jax_out, _ = _rewrite(jnp.array(grid), tuple(map(jnp.array, wb)),
                                                   tuple(map(jnp.array, lb)), random.PRNGKey(0), params)
                            np.testing.assert_array_equal(jax_out, expected)
                            checks += 1
        self.assertEqual(checks, 576)

    def test_archived_corner_mode_preserves_original_scatter(self):
        grid = np.zeros((8, 8), np.int32)
        grid[0, 0], grid[4, 4] = 1, 17
        source = (jnp.array([4, 4]), jnp.array([4, 4]))
        target = (jnp.array([0, 0]), jnp.array([0, 0]))
        for legacy in (False, True):
            params = SCHCParams(L=8, mu=0, death_prob=0, legacy_corner=legacy)
            got, _ = _rewrite(jnp.array(grid), source, target, random.PRNGKey(0), params)
            self.assertEqual(int(got[0, 0]), 1 if legacy else 17)

    def test_dyadic_winner_does_not_depend_on_sampling_order(self):
        for a in range(1, 16):
            grid = np.zeros((8, 8), np.int32)
            grid[1, 1], grid[5, 5] = a, 731
            g = jnp.array(grid)
            first = jnp.zeros_like(g, dtype=bool).at[1, 1].set(True)
            second = jnp.zeros_like(g, dtype=bool).at[5, 5].set(True)
            forward = int(_dyadic_winner(g, first, second, "intended"))
            backward = int(_dyadic_winner(g, second, first, "intended"))
            self.assertEqual(forward, 1 - backward)

    def test_cpu_replays_archived_prefix(self):
        path = ROOT / "results/cpu_scan_full/original_mix/L100/runs/seed_0.csv"
        if not path.exists():
            self.skipTest("extract the data archive for the archived-prefix check")
        ref = np.genfromtxt(path, delimiter=",", names=True)
        rows, _ = fast.run(100, 0, 100, sample_interval=10)
        expected = np.column_stack([ref[c][:len(rows)] for c in fast.COLUMNS])
        # The archived runner serializes score/size columns to eight decimals.
        np.testing.assert_array_equal(np.round(rows, 8), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
