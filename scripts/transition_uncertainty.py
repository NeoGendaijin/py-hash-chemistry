#!/usr/bin/env python3
"""Uncertainty estimates for the SCHC size scan (npj Complexity revision, R1.1).

For every grid size L of the coarse scan (results/transition_scan) this computes
the Table 2 quantities together with 95% intervals:

  * runaway fraction k/n (final mean component size > 100) with a Wilson interval;
  * final mean and maximum component size: ensemble mean with a percentile
    bootstrap interval over runs, plus the median and interquartile range;
  * Late/Early ratio of the ensemble-mean size trajectory
    (steps 15,000-20,000 over 2,000-5,000) with a bootstrap interval;
  * Pearson r between the ensemble-mean size and hash-score trajectories over
    steps 2,000-10,000 with a bootstrap interval.

Bootstrap resamples whole runs (n=10 per L) with replacement, 10,000 times,
with a fixed seed. Point estimates are checked against the values already
reported in results/transition_scan/analysis/transition_analysis.csv.

The fine scan (results/fine_transition_scan, L=300..320) gets runaway fractions
with Wilson intervals and an exact permutation test of homogeneity across L.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COARSE = ROOT / "results" / "transition_scan"
FINE = ROOT / "results" / "fine_transition_scan"
OUT = ROOT / "results" / "statistical_analysis"

RUNAWAY = 100.0
B = 10_000
SEED = 20260925
EARLY, LATE, CORR = (2000, 5000), (15000, 20000), (2000, 10000)
COLS = ["step", "mean_size", "max_size", "mean_fitness"]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def load_runs(d: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dfs = [pd.read_csv(f, usecols=COLS) for f in sorted((d / "runs").glob("seed_*.csv"))]
    lengths = {len(df) for df in dfs}
    if len(lengths) != 1:
        raise ValueError(f"{d}: runs have different lengths {sorted(lengths)}")
    steps = dfs[0]["step"].to_numpy()
    for df in dfs[1:]:
        if not np.array_equal(df["step"].to_numpy(), steps):
            raise ValueError(f"{d}: runs are sampled on different step grids")
    mats = {c: np.stack([df[c].to_numpy(float) for df in dfs]) for c in COLS[1:]}
    return steps, mats


def window(steps: np.ndarray, w: tuple[int, int]) -> np.ndarray:
    return (steps >= w[0]) & (steps <= w[1])


def rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    den = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x * y).sum(axis=1) / den


def pct(a: np.ndarray) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def analyse_coarse(L: int, rng: np.random.Generator) -> dict:
    steps, m = load_runs(COARSE / f"L{L}")
    n = m["mean_size"].shape[0]
    fin_mean, fin_max = m["mean_size"][:, -1], m["max_size"][:, -1]
    early = m["mean_size"][:, window(steps, EARLY)].mean(axis=1)
    late = m["mean_size"][:, window(steps, LATE)].mean(axis=1)
    cw = window(steps, CORR)
    S, Q = m["mean_size"][:, cw], m["mean_fitness"][:, cw]

    # resampling weights: row b gives how often each run is drawn / n
    idx = rng.integers(0, n, size=(B, n))
    W = np.stack([np.bincount(r, minlength=n) for r in idx]).astype(float) / n
    boot_r = np.concatenate([rowwise_pearson(W[i:i + 500] @ S, W[i:i + 500] @ Q)
                             for i in range(0, B, 500)])
    k = int((fin_mean > RUNAWAY).sum())
    lo, hi = wilson(k, n)
    q_mean = np.percentile(fin_mean, [25, 50, 75])
    q_max = np.percentile(fin_max, [25, 50, 75])
    return {
        "L": L, "n": n, "runaway_k": k, "runaway_frac": k / n,
        "runaway_lo": lo, "runaway_hi": hi,
        "mean_size": fin_mean.mean(), "mean_size_lo": pct(W @ fin_mean)[0],
        "mean_size_hi": pct(W @ fin_mean)[1],
        "mean_size_q25": q_mean[0], "mean_size_median": q_mean[1], "mean_size_q75": q_mean[2],
        "max_size": fin_max.mean(), "max_size_lo": pct(W @ fin_max)[0],
        "max_size_hi": pct(W @ fin_max)[1],
        "max_size_q25": q_max[0], "max_size_median": q_max[1], "max_size_q75": q_max[2],
        "late_early": late.mean() / early.mean(),
        "late_early_lo": pct((W @ late) / (W @ early))[0],
        "late_early_hi": pct((W @ late) / (W @ early))[1],
        "r": float(rowwise_pearson(S.mean(axis=0)[None], Q.mean(axis=0)[None])[0]),
        "r_lo": pct(boot_r)[0], "r_hi": pct(boot_r)[1],
    }


def perm_homogeneity(ks: list[int], ns: list[int], rng: np.random.Generator,
                     n_perm: int = 100_000) -> tuple[float, float]:
    """Permutation test that runaway probability is the same at every L (chi-square statistic)."""
    labels = np.repeat(np.arange(len(ns)), ns)
    outcome = np.concatenate([[1] * k + [0] * (n - k) for k, n in zip(ks, ns)])
    p0 = outcome.mean()
    exp1 = np.array(ns) * p0
    exp0 = np.array(ns) * (1 - p0)

    def chi2(o: np.ndarray) -> float:
        o1 = np.bincount(labels, weights=o, minlength=len(ns))
        o0 = np.array(ns) - o1
        return float(((o1 - exp1) ** 2 / exp1 + (o0 - exp0) ** 2 / exp0).sum())

    obs = chi2(outcome)
    hits = sum(chi2(rng.permutation(outcome)) >= obs - 1e-12 for _ in range(n_perm))
    return obs, (hits + 1) / (n_perm + 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = [analyse_coarse(L, rng) for L in range(200, 401, 20)]
    coarse = pd.DataFrame(rows)
    coarse.to_csv(OUT / "transition_uncertainty.csv", index=False)

    # consistency with the numbers already in the paper
    ref = pd.read_csv(COARSE / "analysis" / "transition_analysis.csv").set_index("L")
    chk = pd.DataFrame({
        "mean_size": coarse.set_index("L")["mean_size"] - ref["mean_size_end"],
        "max_size": coarse.set_index("L")["max_size"] - ref["max_size_end"],
        "late_early": coarse.set_index("L")["late_early"] - ref["late_over_early"],
        "r": coarse.set_index("L")["r"] - ref["corr_size_fitness_2k_10k"],
    })
    print("max |difference| vs transition_analysis.csv:")
    print(chk.abs().max().to_string())

    fine_rows = []
    for L in range(300, 321, 2):
        steps, m = load_runs(FINE / f"L{L}")
        fin = m["mean_size"][:, -1]
        k, n = int((fin > RUNAWAY).sum()), len(fin)
        lo, hi = wilson(k, n)
        fine_rows.append({"L": L, "n": n, "runaway_k": k, "runaway_frac": k / n,
                          "runaway_lo": lo, "runaway_hi": hi, "final_step": int(steps[-1])})
    fine = pd.DataFrame(fine_rows)
    fine.to_csv(OUT / "fine_transition_uncertainty.csv", index=False)

    sub = fine[fine["L"] >= 302]
    stat, p = perm_homogeneity(sub["runaway_k"].tolist(), sub["n"].tolist(), rng)
    pooled = sub["runaway_k"].sum() / sub["n"].sum()
    plo, phi = wilson(int(sub["runaway_k"].sum()), int(sub["n"].sum()))
    with open(OUT / "fine_transition_homogeneity.txt", "w") as fh:
        fh.write(f"L=302..320: pooled runaway {sub['runaway_k'].sum()}/{sub['n'].sum()}"
                 f" = {pooled:.3f} (Wilson 95% {plo:.3f}-{phi:.3f})\n"
                 f"permutation chi-square homogeneity test: chi2={stat:.2f}, p={p:.4f}\n")

    pd.set_option("display.width", 250)
    print("\ncoarse scan:")
    print(coarse.round(3).to_string(index=False))
    print("\nfine scan:")
    print(fine.round(3).to_string(index=False))
    print(open(OUT / "fine_transition_homogeneity.txt").read())


if __name__ == "__main__":
    main()
