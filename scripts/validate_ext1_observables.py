#!/usr/bin/env python3
"""Regression checks for separate Extension I observables; no simulations required."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import npj_rev_ext1 as ext1
from verify_npj_revision import REMOVED_COMPOSITES, separate_observables


class SeparateObservables(unittest.TestCase):
    def test_condition_summary_and_bootstraps_have_no_composite(self):
        runs = pd.DataFrame({"rule": ["intended"] * 4, "D": [100] * 4, "P": [0.0] * 4,
                             "seed": range(4), "used": [True, True, True, False],
                             "B_run": [2.0, 4.0, 6.0, 100.0], "C_run": [10.0, 20.0, 30.0, 1000.0],
                             "category": ["success"] * 3 + ["no_growth"]})
        curves = {("intended", 100, 0.0, seed): np.full(5, value)
                  for seed, value in enumerate(runs.B_run)}
        table, boots = ext1.condition_table(runs, curves, 1000, np.random.default_rng(7))
        self.assertFalse(REMOVED_COMPOSITES.intersection(table.columns))
        np.testing.assert_allclose(table[["A", "B", "C"]].iloc[0], [3, 4, 20])
        self.assertEqual(set(boots[("intended", 100, 0.0)]), {"A", "B", "C", "logC"})
        self.assertLess(table.A_wilson_lo.iloc[0], 3)
        self.assertGreater(table.A_wilson_hi.iloc[0], 3)

    def test_three_heatmaps_and_snapshot_labels(self):
        self.assertEqual([spec[0] for spec in ext1.HEAT_SPECS], ["A", "B", "C"])
        self.assertEqual(set(ext1.label_set()), {(D, P) for D, P, _ in ext1.SNAP_CONDS})
        self.assertFalse(any("max" in label for label in ext1.label_set().values()))

    def test_pair_plots_use_the_parameter_P_without_composite_columns(self):
        table = pd.DataFrame([dict(rule="intended", D=D, P=P, A=23 + 7 * P,
                                   B=5 - P + D / 1000, C=7000 + 30000 * P + D)
                              for D in ext1.D_VALUES for P in ext1.P_VALUES])
        saved = []

        def inspect(fig, out, name):
            axes = fig.axes[:3] if name == "fig6_pairs" else fig.axes[:1]
            for ax in axes:
                colors = np.concatenate([np.asarray(c.get_array()) for c in ax.collections
                                         if c.get_array() is not None])
                np.testing.assert_array_equal(np.sort(colors), np.sort(table.P))
            self.assertEqual(fig.axes[-1].get_ylabel(), "Dyadic interaction probability $P$")
            saved.append(name)
            plt.close(fig)

        with patch.object(ext1, "savefig", side_effect=inspect):
            ext1.fig6(table, Path("unused"))
        self.assertEqual(saved, ["fig6_3d", "fig6_pairs"])

    def test_heatmap_layout_and_outputs_have_only_three_observables(self):
        table = pd.DataFrame([dict(rule="intended", D=D, P=P, A=23 + 7 * P,
                                   B=5 - P + D / 1000, C=7000 + 30000 * P + D)
                              for D in ext1.D_VALUES for P in ext1.P_VALUES])
        saved = []

        def inspect(fig, out, name):
            if name == "fig5_heatmaps":
                self.assertEqual(len(fig.axes), 6)
                panels = fig.axes[::2]
                self.assertTrue(all(ax.get_title(loc="left").startswith(f"({label})")
                                    for ax, label in zip(panels, "abc")))
                self.assertEqual(panels[1].get_ylabel(), "")
            saved.append(name)
            plt.close(fig)

        with patch.object(ext1, "savefig", side_effect=inspect):
            ext1.fig5(table, Path("unused"))
        self.assertEqual(saved, ["fig5_heatmaps", *(spec[1] for spec in ext1.HEAT_SPECS)])

    def test_manuscript_guard_rejects_removed_claims_and_formulas(self):
        for text in ("overall performance", "largest-product condition", r"A$\times$B$\times$C", "A*B*C"):
            with self.subTest(text=text), self.assertRaises(AssertionError):
                separate_observables(text)
        separate_observables("A, B and C are reported separately; color denotes P.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
