#!/usr/bin/env python3
"""Check manuscript limits, figure inputs and table values against archived runs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SIZES = [100, 120, 140, 160, 180, 200, 210, 220, 230, 240, 250,
         260, 280, 300, 320, 340, 360, 380, 400]
REMOVED_COMPOSITES = {"ABC", "ABC_lo", "ABC_hi", "ABC_rel", "rank_ABC",
                      "composite_rank", "ABgmC", "rank_ABgmC"}


def separate_observables(text):
    assert not re.search(r"overall[- ]performance|largest[- ]product|rank_ABC", text, re.I), \
        "removed aggregate claim is present"
    compact = re.sub(r"[\s${}]", "", text)
    assert "A\\timesB\\timesC" not in compact and "A*B*C" not in compact, \
        "removed aggregate formula is present"


def argument(text, position):
    start = text.index("{", position)
    depth = 1
    i = start + 1
    while depth:
        if text[i] == "\\":
            i += 2
            continue
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return text[start + 1:i - 1], i


def words(text):
    text = re.sub(r"\\(?:rev|moved)(?:\[[^\]]*\])?", "", text)
    text = re.sub(r"\\(?:cite\w*|ref|label|url)\{[^{}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}$~]", " ", text)
    return len(text.split())


def manuscript_checks(tex):
    text = tex.read_text()
    # Remove comments without swallowing escaped percent signs.
    active = "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in text.splitlines())
    separate_observables(active)
    supplement = tex.parent / "supplementary-information.tex"
    if supplement.exists():
        separate_observables(supplement.read_text())
    assert not re.search(r"\\todo\s*\{", active), "manuscript still contains a TODO"
    abstract, _ = argument(active, active.index("\\abstract"))
    nw = words(abstract)
    assert nw <= 150, f"abstract: {nw} words (limit 150)"
    captions = []
    for i, m in enumerate(re.finditer(r"\\begin\{figure\}(.*?)(?:\\end\{figure\})", active, re.S), 1):
        body = m.group(1)
        cap, _ = argument(body, body.index("\\caption"))
        count = words(cap)
        captions.append(count)
        assert count <= 350, f"Figure {i} legend: {count} words (limit 350)"
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]*)\}", active):
        assert (tex.parent / fig).is_file(), f"missing manuscript figure: {fig}"
    assert "\\section{Conclusion}" not in active
    expected = (ROOT / "results/npj_rev/ext2_tables/table2_valci.tex").read_text().splitlines()
    for row in expected:
        if re.match(r"^\s*\d+\s*&", row):
            assert row.strip() in text, f"Table 2 row differs from generated values: {row[:40]}"
    return {"abstract_words": nw, "figure_legend_words": captions, "todos": 0,
            "extension_I_separate_observables": True}


def matched_seed_checks(intended):
    reported = pd.read_csv(ROOT / "results/npj_rev/supplement/matched_seed_checks.csv")
    assert len(reported) == 2 and set(reported.measure) == {"B_run", "C_run"}
    assert reported.comparison.str.startswith("P=1 minus P=0").all()
    ok = intended[intended.used]
    for row in reported.itertuples():
        columns = ["D", "seed", row.measure]
        pairs = ok[ok.P == 0][columns].merge(ok[ok.P == 1][columns], on=["D", "seed"],
                                            suffixes=("_0", "_1"), validate="one_to_one")
        pairs["difference"] = pairs[row.measure + "_1"] - pairs[row.measure + "_0"]
        differences = pairs.groupby("seed").difference.mean().to_numpy()
        assert row.n == len(differences) == 30
        np.testing.assert_allclose(row.mean_difference, differences.mean(), rtol=1e-12)
        np.testing.assert_allclose(row.p, stats.wilcoxon(differences).pvalue, rtol=1e-12)
        rng = np.random.default_rng(int(row.bootstrap_seed))
        samples = differences[rng.integers(0, len(differences), (10000, len(differences)))].mean(axis=1)
        np.testing.assert_allclose([row.ci_lo, row.ci_hi], np.percentile(samples, [2.5, 97.5]), rtol=1e-12)
    return len(reported)


def data_checks():
    table = pd.read_csv(ROOT / "results/npj_rev/ext2_tables/table2.csv").set_index("L")
    counts = {}
    reversals = 0
    for L in SIZES:
        runaways = 0
        means, maxima = [], []
        for seed in range(100):
            path = ROOT / f"results/cpu_scan_full/original_mix/L{L}/runs/seed_{seed}.csv"
            d = pd.read_csv(path)
            assert d.step.iloc[0] == 0
            assert d.step.iloc[-1] == 20000 or d.n_active.iloc[-1] == 0, path
            assert d.step.is_monotonic_increasing and not d.step.duplicated().any(), path
            assert (d.cum_pattern_types.diff().dropna() >= 0).all(), path
            crossing = np.flatnonzero(d.mean_size.to_numpy() > 100)
            if len(crossing):
                reversals += int((d.mean_size.iloc[crossing[0]:] <= 100).any())
            runaways += bool(d.mean_size.iloc[-1] > 100)
            means.append(d.mean_size.iloc[-1])
            maxima.append(d.max_size.iloc[-1])
        counts[L] = int(runaways)
        if L in table.index:
            t = table.loc[L]
            assert int(t.runaway_n) == 100 and int(t.runaway_k) == runaways
            np.testing.assert_allclose([np.mean(means), np.mean(maxima)],
                                       [t.mean_size, t.max_size], rtol=1e-12)
    runs = pd.read_csv(ROOT / "results/npj_rev/ext1/per_run.csv")
    intended = runs[runs.rule == "intended"]
    conditions = pd.read_csv(ROOT / "results/npj_rev/ext1/conditions.csv")
    assert not REMOVED_COMPOSITES.intersection(conditions.columns), "obsolete composite columns"
    assert len(intended) == 1800
    assert (intended.groupby(["D", "P"]).size() == 30).all()
    for row in intended.itertuples():
        base = ROOT / f"results/corner_fixed/spatial_dyadic/intended/D{row.D}_P{row.P:.1f}"
        meta = json.loads((base / "run_metadata.json").read_text())
        assert meta["definition"]["model"]["legacy_corner"] is False
        assert meta["definition"]["model"]["dyadic_rule"] == "intended"
        raw = pd.read_csv(base / f"seed_{row.seed}.csv")
        assert len(raw) == 2000 and raw.step.iloc[-1] == 1999
        np.testing.assert_allclose(raw.cum_pattern_types.iloc[-1], row.C_run)
        np.testing.assert_allclose(raw.mean_rep_size.iloc[-500:].mean(), row.B_run)
        assert bool(raw.max_rep_size.max() >= 1000) == bool(row.runaway)
    assert reversals == 0
    return {"size_scan_runs": 1900, "runaway_by_size": counts,
            "sampled_reversals": reversals, "extension_I_runs": len(intended),
            "extension_I_successes": int(intended.used.sum()),
            "extension_I_runaways": int(intended.runaway.sum()),
            "extension_I_legacy_corner": False,
            "extension_I_matched_dyadicity_checks": matched_seed_checks(intended)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", type=Path, default=ROOT / "tex/Large-Hash-Chemistry/sn-article-rev1.tex")
    args = ap.parse_args()
    result = {"data": data_checks()}
    if args.tex.exists():
        result["manuscript"] = manuscript_checks(args.tex)
    else:
        result["manuscript"] = "not present in code checkout; data checks completed"
    out = ROOT / "results/npj_rev/verification.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
