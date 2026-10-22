#!/usr/bin/env python3
"""Check manuscript limits, figure inputs and table values against archived runs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIZES = [100, 120, 140, 160, 180, 200, 210, 220, 230, 240, 250,
         260, 280, 300, 320, 340, 360, 380, 400]


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
    return {"abstract_words": nw, "figure_legend_words": captions, "todos": 0}


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
    assert len(intended) == 1800
    assert (intended.groupby(["D", "P"]).size() == 30).all()
    assert int(intended.used.sum()) == 1711
    assert int((intended.category == "runaway").sum()) == 3
    assert reversals == 0
    return {"size_scan_runs": 1900, "runaway_by_size": counts,
            "sampled_reversals": reversals, "extension_I_runs": len(intended),
            "extension_I_successes": int(intended.used.sum())}


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
