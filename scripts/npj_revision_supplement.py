#!/usr/bin/env python3
"""Additional observables and matched-seed checks from existing revision data."""
from __future__ import annotations

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/npj_rev/supplement"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def spatial_observables():
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.2), layout="constrained")
    metrics = [("occupancy", "Active cells / grid area"),
               ("dominant_share", "Largest component / active cells"),
               ("n_components", "Number of components")]
    colors = {200: "#0072B2", 240: "#D55E00", 400: "#009E73"}
    grid = np.arange(0, 20001, 10)
    rows = []
    for L in (200, 240, 400):
        groups = {False: [], True: []}
        for seed in range(100):
            path = ROOT / f"results/cpu_scan_full/original_mix/L{L}/runs/seed_{seed}.csv"
            d = pd.read_csv(path)
            runaway = bool(d.mean_size.iloc[-1] > 100)
            x = d.set_index("step").reindex(grid).fillna(0)
            x["occupancy"] = x.n_active / L**2
            x["dominant_share"] = np.divide(x.max_size, x.n_active,
                                             out=np.zeros(len(x)), where=x.n_active > 0)
            groups[runaway].append(x[[m for m, _ in metrics]].to_numpy())
            rows.append(dict(L=L, seed=seed, runaway=runaway,
                             occupancy=x.occupancy.iloc[-1], dominant_share=x.dominant_share.iloc[-1],
                             n_components=x.n_components.iloc[-1]))
        for runaway, members in groups.items():
            arr = np.stack(members)
            low, med, high = np.percentile(arr, [25, 50, 75], axis=0)
            label = f"L={L}, {'runaway' if runaway else 'compact/extinct'} (n={len(arr)})"
            style = "-" if runaway else "--"
            for k, ax in enumerate(axs):
                ax.plot(grid, med[:, k], color=colors[L], ls=style, lw=1.2, label=label)
                ax.fill_between(grid, low[:, k], high[:, k], color=colors[L], alpha=0.07)
    for k, (ax, (_, label)) in enumerate(zip(axs, metrics)):
        ax.set(xlabel="Step", ylabel=label, title=f"({chr(97+k)})")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4))
        ax.spines[["top", "right"]].set_visible(False)
    axs[2].legend(fontsize=6.2, frameon=False)
    save(fig, "spatial_observables")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "spatial_observables.csv", index=False)
    return out


def novelty_rate():
    rates = []
    bins = np.arange(0, 20001, 1000)
    for seed in range(40):
        d = pd.read_csv(ROOT / f"results/cpu_novelty/original_mix/L200/runs/seed_{seed}.csv")
        s = d.set_index("step").cum_pattern_types.reindex(bins).ffill()
        for t, rate in zip(bins[1:], np.diff(s.to_numpy()) / 1000):
            rates.append(dict(seed=seed, window_end=t, novel_patterns_per_step=rate))
    out = pd.DataFrame(rates)
    out.to_csv(OUT / "novelty_rate.csv", index=False)
    q = out.groupby("window_end").novel_patterns_per_step.quantile([0.25, 0.5, 0.75]).unstack()
    fig, ax = plt.subplots(figsize=(5.4, 2.8), layout="constrained")
    ax.plot(q.index, q[0.5], color="#0072B2", lw=1.4)
    ax.fill_between(q.index, q[0.25], q[0.75], color="#0072B2", alpha=0.18)
    ax.set(xlabel="End of 1,000-step window", ylabel="New sampled pattern types per step")
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "novelty_rate")
    return out


def matched_seed_checks():
    d = pd.read_csv(ROOT / "results/npj_rev/ext1/per_run.csv")
    d = d[d.rule == "intended"].copy()
    original = d[(d.D == 100) & (d.P == 0)].set_index("seed")
    best = d[(d.D == 100) & (d.P == 1)].set_index("seed")
    rng = np.random.default_rng(20261023)
    records = []
    orig_used = original.used.to_numpy(bool)
    best_used = best.loc[original.index].used.to_numpy(bool)
    gains, losses = int((~orig_used & best_used).sum()), int((orig_used & ~best_used).sum())
    p = stats.binomtest(gains, gains + losses).pvalue if gains + losses else 1.0
    records.append(dict(comparison="(100,1) minus (100,0)", measure="success", n=30,
                        mean_difference=float((best_used.astype(int) - orig_used).mean()),
                        ci_lo=np.nan, ci_hi=np.nan, p=p,
                        test="exact paired McNemar (discordant seeds)", gains=gains, losses=losses))
    for col in ("B_run", "C_run"):
        mask = original.used & best.used
        diff = best.loc[mask, col].to_numpy() - original.loc[mask, col].to_numpy()
        boots = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        records.append(dict(comparison="(100,1) minus (100,0)", measure=col, n=len(diff),
                            mean_difference=diff.mean(), ci_lo=lo, ci_hi=hi,
                            p=stats.wilcoxon(diff).pvalue, test="paired Wilcoxon; jointly successful seeds",
                            gains=np.nan, losses=np.nan))
    for col in ("B_run", "C_run"):
        ok = d[d.used]
        a = ok[ok.P == 0][["D", "seed", col]]
        b = ok[ok.P == 1][["D", "seed", col]]
        pairs = a.merge(b, on=["D", "seed"], suffixes=("_0", "_1"))
        pairs["difference"] = pairs[f"{col}_1"] - pairs[f"{col}_0"]
        diff = pairs.groupby("seed").difference.mean().to_numpy()
        boots = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        records.append(dict(comparison="P=1 minus P=0, matched D and seed; averaged within seed",
                            measure=col, n=len(diff), mean_difference=diff.mean(), ci_lo=lo, ci_hi=hi,
                            p=stats.wilcoxon(diff).pvalue, test="paired Wilcoxon across seed blocks",
                            gains=np.nan, losses=np.nan))
    out = pd.DataFrame(records)
    out.to_csv(OUT / "matched_seed_checks.csv", index=False)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    spatial = spatial_observables()
    novelty = novelty_rate()
    paired = matched_seed_checks()
    lines = ["# Additional checks from existing trajectories", "",
             "No new simulation runs were generated for these checks.", "",
             "## Final spatial observables (median)", "",
             spatial.groupby(["L", "runaway"])[["occupancy", "dominant_share", "n_components"]]
                    .median().to_string(), "", "## Late novelty rate", ""]
    late = novelty[novelty.window_end > 15000].groupby("seed").novel_patterns_per_step.mean()
    lines += [f"L=200, steps 15,001-20,000, 40 runs: median {late.median():.3f} new sampled pattern "
              f"types/step, IQR {late.quantile(.25):.3f}-{late.quantile(.75):.3f}.",
              "Counts refer to sampled components, not exclusively replicated winners.", "",
              "## Matched-seed sensitivity checks", "", paired.to_string(index=False), "",
              "These exploratory contrasts preserve seed dependence across conditions. B/C comparisons "
              "are conditional on success in both matched conditions. Their intervals resample seeds, "
              "not individual contests or time points. They do not convert the product ranking into a "
              "pre-registered optimum or establish equivalence of implementations."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
