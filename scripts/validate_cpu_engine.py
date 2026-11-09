#!/usr/bin/env python3
"""
Distributional validation of the CPU engine (src/schc_fast.c) against the JAX runs.

  legacy/xor (published rule): results/cpu_scan/legacy_xor/L{200,400}  vs  results/large_space/L{200,400}
      (100 JAX runs each, used in the npj paper)
  original/mix (corrected):    results/cpu_scan/original_mix/L200      vs  results/transition_fixed/L200

Compared per L: runaway fraction (final mean component size > 100; Fisher exact),
onset step of runaway runs (first sample with mean size > 100; Mann-Whitney),
and, among non-runaway runs, final mean size, final max size and final occupancy (Mann-Whitney).
Final values are taken at step 20000 (or at the early stop of runaway runs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ("current corner-corrected original/mix L=100", ROOT / "results/cpu_scan_check/original_mix/L100/runs",
     ROOT / "results/corner_fixed/transition_fixed_check/L100/runs", 100),
    ("legacy/xor L=200", ROOT / "results/cpu_scan/legacy_xor/L200/runs", ROOT / "results/large_space/L200/runs", 200),
    ("legacy/xor L=400", ROOT / "results/cpu_scan/legacy_xor/L400/runs", ROOT / "results/large_space/L400/runs", 400),
    ("original/mix L=200", ROOT / "results/cpu_scan/original_mix/L200/runs", ROOT / "results/transition_fixed/L200/runs", 200),
    ("original/mix L=320", ROOT / "results/cpu_scan_full/original_mix/L320/runs", ROOT / "results/transition_fixed/L320/runs", 320),
    ("original/mix L=100", ROOT / "results/cpu_scan_check/original_mix/L100/runs", ROOT / "results/transition_fixed_check/L100/runs", 100),
]


def summarize(run_dir: Path, L: int):
    out = []
    for f in sorted(run_dir.glob("seed_*.csv")):
        if f.name.endswith(".partial.csv"):
            continue
        d = pd.read_csv(f)
        last = d.iloc[-1]
        over = d[d["mean_size"] > 100]
        dens = last["n_active"] / L**2 if "n_active" in d else np.nan
        out.append(dict(runaway=bool(last["mean_size"] > 100), onset=float(over["step"].iloc[0]) if len(over) else np.nan,
                        mean_size=float(last["mean_size"]), max_size=float(last["max_size"]), density=dens))
    return pd.DataFrame(out)


def main():
    lines = ["# CPU engine vs JAX: distributional validation\n",
             "The current comparison uses corner-corrected JAX. The remaining comparisons below use archived JAX runs "
             "with the historical corner behavior and are retained as historical checks, not new corrected runs.\n"]
    for name, cpu_dir, jax_dir, L in PAIRS:
        if not cpu_dir.exists() or not jax_dir.exists():
            lines.append(f"## {name}: missing data\n")
            continue
        c, j = summarize(cpu_dir, L), summarize(jax_dir, L)
        kc, kj = int(c.runaway.sum()), int(j.runaway.sum())
        p_run = stats.fisher_exact([[kc, len(c) - kc], [kj, len(j) - kj]])[1]
        lines.append(f"## {name}\n")
        lines.append(f"- runaway: CPU {kc}/{len(c)}, JAX {kj}/{len(j)} (Fisher p={p_run:.3g})")
        if kc > 2 and kj > 2:
            p = stats.mannwhitneyu(c.onset.dropna(), j.onset.dropna()).pvalue
            lines.append(f"- onset step of runaway runs: CPU median {c.onset.median():.0f}, JAX median {j.onset.median():.0f} (MWU p={p:.3g})")
        cn, jn = c[~c.runaway], j[~j.runaway]
        for col in ("mean_size", "max_size", "density"):
            if cn[col].notna().sum() > 2 and jn[col].notna().sum() > 2:
                p = stats.mannwhitneyu(cn[col].dropna(), jn[col].dropna()).pvalue
                lines.append(f"- non-runaway final {col}: CPU median {cn[col].median():.3g} [IQR {cn[col].quantile(.25):.3g}-{cn[col].quantile(.75):.3g}], "
                             f"JAX median {jn[col].median():.3g} [IQR {jn[col].quantile(.25):.3g}-{jn[col].quantile(.75):.3g}] (MWU p={p:.3g})")
        lines.append("")
    text = "\n".join(lines)
    out = ROOT / "results/cpu_scan/validation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
