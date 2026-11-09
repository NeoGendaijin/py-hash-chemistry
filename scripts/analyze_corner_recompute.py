#!/usr/bin/env python3
"""Audit corrected run provenance and quantify changes from the corner-legacy sweep."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "results/corner_fixed"
OUT = ROOT / "results/npj_rev/corner_check"


def audit_metadata():
    paths = sorted(NEW.rglob("run_metadata.json"))
    assert len(paths) == 118, (len(paths), "expected 110 sweep, 4 snapshot, 3 comparison, 1 validation receipts")
    for path in paths:
        metadata = json.loads(path.read_text())
        assert metadata["definition"]["model"]["legacy_corner"] is False, path
        for name, digest in metadata["source_sha256"].items():
            candidates = [ROOT / name, *(NEW / "source_snapshot").rglob(Path(name).name)]
            assert any(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == digest for p in candidates), (path, name)
    for name in ("sweep_part_0", "sweep_part_1", "sweep_part_2", "original_legacy", "original_fixed",
                 "original_fixed_mix", "snapshots", "validation_L100"):
        completion = json.loads((NEW / f"{name}.completion.json").read_text())
        assert completion["returncode"] == 0, name
        assert completion["command"][:4] == ["uv", "run", "--no-sync", "python"], name
    assert not (NEW / "THERMAL_STOP").exists()
    return len(paths)


def main():
    count = audit_metadata()
    exact_columns = ["step", "num_replicated_cells", "max_rep_size", "mean_rep_size", "cum_cell_types",
                     "cum_pattern_types", "n_events", "n_active", "log_bytes_cum"]
    for seed in range(30):
        a = pd.read_csv(NEW / f"spatial_dyadic/intended/D100_P0.0/seed_{seed}.csv")
        b = pd.read_csv(NEW / f"original_comparison/fixed_mix/seed_{seed}.csv")
        # D=None and D=L both take the same global-sampling branch. These
        # independently run datasets use batches of 50 and 30 respectively.
        np.testing.assert_array_equal(a[exact_columns].to_numpy(), b[exact_columns].to_numpy())
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-7, atol=1e-10, equal_nan=True)
    out = pd.read_csv(ROOT / "results/npj_rev/ext1/per_run.csv")
    old_path = ROOT / "results/npj_rev/corner_check/previous_per_run.csv"
    old = pd.read_csv(old_path)
    d = out.merge(old, on=["rule", "D", "P", "seed"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(d) == 3600
    d.to_csv(OUT / "paired_old_new.csv", index=False)
    rows = []
    for rule, g in d.groupby("rule"):
        for col in ("B_run", "C_run"):
            both = g.used_new & g.used_old
            x, y = g.loc[both, col + "_new"], g.loc[both, col + "_old"]
            delta = x - y
            blocks = pd.DataFrame({"seed": g.loc[both, "seed"], "difference": delta})
            block_delta = blocks.groupby("seed").difference.mean().to_numpy()
            rows.append(dict(rule=rule, measure=col, jointly_successful=int(both.sum()),
                             old_mean=float(y.mean()), new_mean=float(x.mean()),
                             median_paired_difference=float(delta.median()),
                             changed_runs=int((~np.isclose(x, y, equal_nan=True)).sum()),
                             seed_blocks=len(block_delta),
                             paired_wilcoxon_p=float(stats.wilcoxon(block_delta).pvalue) if np.any(block_delta) else 1.0))
    changes = pd.DataFrame(rows)
    changes.to_csv(OUT / "changes.csv", index=False)
    primary = out[out.rule == "intended"]
    cat = primary.category.value_counts().to_dict()
    original = pd.read_csv(NEW / "original_comparison/analysis/summary.csv")
    comparisons = {}
    for arm, g in original.groupby("arm"):
        raw = g.mwu_p.to_numpy()
        order = np.argsort(raw)
        adjusted = np.minimum(1, np.maximum.accumulate(raw[order] * np.arange(len(raw), 0, -1)))
        comparisons[arm] = dict(raw_significant=int((raw < .05).sum()),
                                holm_significant=int((adjusted < .05).sum()), min_p=float(raw.min()))
    summary = dict(metadata_receipts=count, cross_batch_matched_2000_step_runs=30, extension_I_categories=cat,
                   original_comparison=comparisons, commands_use_uv=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Corner-corrected recalculation audit", "", json.dumps(summary, indent=2), "",
             changes.to_string(index=False), "",
             "Paired old/new contrasts average matched-condition differences within each seed before a Wilcoxon test "
             "across seed blocks. They are exploratory checks, not equivalence tests. A changed corner can alter later "
             "events throughout a trajectory; the correction is not assumed negligible because it affects one site."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
