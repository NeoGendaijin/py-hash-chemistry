#!/usr/bin/env python3
"""
Extension I: is the rise of pattern exploration C with P explained by more replication events?

C (cumulative distinct pattern types in successful replication, t=2000) is divided by the total
number of replication events E of the run (sum of n_events over the 2000 steps) and by the total
number of replicated cells R. Successful runs only (Sayama's criterion plus no runaway event,
as in scripts/npj_rev_ext1.py). Data: the corner-corrected intended-rule sweep.

Output: results/npj_rev/ext1/novelty_per_event.{csv,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "results/corner_fixed/spatial_dyadic/intended"))
    ap.add_argument("--out", default=str(ROOT / "results/npj_rev/ext1"))
    args = ap.parse_args()

    rows = []
    for cond in sorted(Path(args.root).glob("D*_P*")):
        D = int(cond.name.split("_")[0][1:])
        P = float(cond.name.split("_P")[1])
        for f in sorted(cond.glob("seed_*.csv")):
            x = pd.read_csv(f)
            ok = (len(x) == 2000 and 1e6 < x.log_bytes_cum.iloc[-1] < 3e8
                  and x.n_active.iloc[-1] > 0 and x.max_rep_size.max() < 1000)
            if not ok:
                continue
            C = float(x.cum_pattern_types.iloc[-1])
            E = float(x.n_events.sum())
            R = float(x.num_replicated_cells.sum())
            rows.append(dict(D=D, P=P, seed=int(f.stem.split("_")[1]), C=C, events=E,
                             C_per_1000_events=1000 * C / E, C_per_1000_cells=1000 * C / R))
    r = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "novelty_per_event.csv", index=False)

    by_p = r.groupby("P")[["C", "events", "C_per_1000_events", "C_per_1000_cells"]].mean()
    root = Path(args.root).resolve()
    shown = root.relative_to(ROOT) if root.is_relative_to(ROOT) else root.name
    lines = ["# Extension I: distinct pattern types per replication event\n",
             f"Successful runs: {len(r)} (intended rule, corner-corrected data, {shown}).\n",
             "| P | C | replication events | C per 1,000 events | C per 1,000 replicated cells |",
             "|---|---|---|---|---|"]
    for P, v in by_p.iterrows():
        lines.append(f"| {P:.1f} | {v.C:,.0f} | {v.events:,.0f} | {v.C_per_1000_events:.1f} | {v.C_per_1000_cells:.2f} |")
    lines.append("")
    for col in ["C", "events", "C_per_1000_events", "C_per_1000_cells"]:
        rho, p = stats.spearmanr(r.P, r[col])
        lines.append(f"- Spearman rho({col}, P) = {rho:.2f}, p = {p:.1e} (n = {len(r)} runs)")
    strata = []
    for D, s in r.groupby("D"):
        rho, p = stats.spearmanr(s.P, s.C_per_1000_events)
        strata.append((D, rho, p))
    k = sum(1 for _, rho, p in strata if rho > 0 and p < 0.05)
    lines.append(f"- Within each D: C per 1,000 events rises with P (Spearman, p < 0.05) in {k} of {len(strata)} strata; "
                 f"rho range {min(s[1] for s in strata):.2f} to {max(s[1] for s in strata):.2f}")
    ratio_events = by_p.loc[1.0, "events"] / by_p.loc[0.0, "events"]
    ratio_rate = by_p.loc[1.0, "C_per_1000_events"] / by_p.loc[0.0, "C_per_1000_events"]
    lines.append(f"- From P=0 to P=1: events x{ratio_events:.2f}, distinct types per 1,000 events x{ratio_rate:.2f}")
    (out / "novelty_per_event.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
