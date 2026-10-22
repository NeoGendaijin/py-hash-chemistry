#!/usr/bin/env python3
"""
npj Complexity revision -- Extension I (Sec. 3, spatial-dyadic SCHC, L=100) with the INTENDED
dyadic rule (JAX re-run of Sayama's sweep, results/spatial_dyadic/intended; the as-coded rule is
used only for comparison).

Run (single process, ~1-2 min; re-runnable whenever more snapshot data arrive):
    .venv/bin/python scripts/npj_rev_ext1.py [--n-boot 10000] [--out results/npj_rev/ext1]

Definitions (Sayama's criterion; the same as scripts/analyze_spatial_dyadic.py):
  success  = 1 MB < log_bytes_cum[-1] < 300 MB, n_active[-1] > 0 and 2000 rows (t = 0..1999)
  runaway  = max_t max_rep_size >= 1000 cells (one replication event copies >= 10% of the grid);
             a runaway run counts as a failure, as Sayama's explosive run (380 MB log) did
  A = number of successful non-runaway runs (of 30)
  B = mean over the final 500 steps of the ensemble-mean (over successful runs) per-step
      mean_rep_size; per run: nanmean of the last 500 mean_rep_size values
  C = mean over successful runs of cum_pattern_types in the last CSV row (step 1999)
  overall performance = A x B x C (Fig. 5d)
  Time convention (compare_original_schc.run_batch): CSV row t holds n_active after t steps but the
  events (replications, patterns, log bytes) of the (t+1)-th step. So row 1999 gives C after all 2000
  steps (t = 2000, end of run) and rows 1500-1999 cover steps 1501-2000 (t = 1500-2000, as the caption).

Fig. 7 seed choice: candidates are seeds 0-7 that are successful and have >= SNAP_MIN_ACTIVE active
cells at the snapshot time (a grown population); among them the seed whose C is closest to the
condition's median C. Dominance statistics are summarised over the same candidates, per snapshot time.

Outputs in --out (nothing is written into tex/):
  (a) fig5_heatmaps.{pdf,png} + single panels P-D-final-num-runs / -size / -nctypes /
      P-D-overall-performance .{pdf,png} (drop-in names of the manuscript's Fig. 5 files)
  (b) fig6_3d.{pdf,png} (3-D scatter, P-D-3D-performance) and fig6_pairs.{pdf,png} (2-D alternative)
  (c) fig7_snapshots.{pdf,png}, single panels snap_D*_P*_seed*_t*.png, figS_snapshots_all_seeds,
      snapshot_selection.csv, snapshot_stats.csv, snapshot_dominance_by_time.csv
  (d) per_run.csv, conditions.csv, regression.csv, joint_tests.csv, trends.csv, trends_strata.csv,
      contrasts_original_vs_best.csv, contrasts_original_vs_all.csv, contrasts_P0_vs_P.csv,
      contrasts_D100_vs_D.csv, figS_trends.{pdf,png}
  (e) failures.csv, failures_by_P.csv, failures_by_D.csv, sensitivity.csv, figS_failures.{pdf,png}
  (f) summary.md: numbers for the Sec. 3 text, keyed to the sentences of sn-article-rev1.tex
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, hsv_to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d import proj3d
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
SD_ROOT = ROOT / "results" / "spatial_dyadic"
SNAP_ROOT = ROOT / "results" / "spatial_dyadic_snapshots"
TEX = ROOT / "tex" / "Large-Hash-Chemistry" / "sn-article-rev1.tex"  # read only
D_VALUES = list(range(10, 101, 10))
P_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
L, K_TYPES, T_END, N_FINAL = 100, 1000, 2000, 500
LOG_LO, LOG_HI, RUNAWAY_SIZE = 1e6, 3e8, 1000
COLS = ["mean_rep_size", "max_rep_size", "cum_pattern_types", "n_active", "log_bytes_cum"]
# snapshot conditions of Fig. 7 (panel order of the manuscript) and their labels
SNAP_CONDS = [(100, 0.0, "(D, P) = (L, 0)"), (10, 0.0, "(D, P) = (0.1L, 0)"),
              (100, 1.0, "(D, P) = (L, 1)"), (20, 0.2, "(D, P) = (0.2L, 0.2)")]
SNAP_TIMES = [2000, 1750, 1500]  # preference order
SNAP_SEEDS = range(8)
SNAP_MIN_ACTIVE = 500  # a Fig. 7 candidate must have a grown population (successful runs with 26-126 cells exist)
A_JITTER = 0.18        # horizontal jitter of the integer A in fig6_pairs (stated in the figure and summary)
ORIGINAL = (100, 0.0)
CATS = ["success", "extinct", "no_growth", "runaway", "log_ge_300MB", "incomplete"]
CAT_LABEL = {"extinct": "Extinct (no active cell at t=1999)", "no_growth": "No growth (log <= 1 MB)",
             "runaway": "Runaway (a replication event >= 1000 cells)", "log_ge_300MB": "Log >= 300 MB",
             "incomplete": "Fewer than 2000 rows"}

# reuse the design matrix / OLS / logit of the earlier regression script (cross-checked below)
_spec = importlib.util.spec_from_file_location("rsd", ROOT / "scripts" / "regress_spatial_dyadic.py")
rsd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rsd)
TERMS = ["1", "D", "D^2", "P", "P^2", "DP"]

# ---------------- figure style (as scripts/generate_fig_schc_figs.py) ----------------
sns.set_theme(context="paper", style="ticks", font="DejaVu Sans")
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a", "figure.dpi": 300, "savefig.dpi": 300,
})
MM = 1 / 25.4
DOUBLE_COL = 180 * MM
HEAT_CMAP = "flare"          # sequential, perceptually ordered, colour-blind safe (seaborn)
PAL = sns.color_palette("colorblind")
FAIL_COLORS = {"extinct": PAL[7], "no_growth": PAL[0], "runaway": PAL[3], "log_ge_300MB": PAL[4],
               "incomplete": PAL[5]}


def savefig(fig, out, name, png_dpi=300):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)


# =============================== data ===============================
def load_rule(rule: str):
    """Per-run records and final-500 mean_rep_size curves of one dyadic rule (P=0 is shared)."""
    recs, curves = [], {}
    for D in D_VALUES:
        for P in P_VALUES:
            src = "intended" if P == 0.0 else rule
            d = SD_ROOT / src / f"D{D}_P{P:.1f}"
            for f in sorted(d.glob("seed_*.csv"), key=lambda p: int(p.stem.split("_")[1])):
                x = pd.read_csv(f, usecols=COLS)
                seed = int(f.stem.split("_")[1])
                log, act = float(x.log_bytes_cum.iloc[-1]), int(x.n_active.iloc[-1])
                maxrep = float(np.nanmax(x.max_rep_size.to_numpy()))
                full = len(x) == T_END
                ok = bool(LOG_LO < log < LOG_HI and act > 0 and full)
                run = bool(maxrep >= RUNAWAY_SIZE)
                if not full:
                    cat = "incomplete"
                elif act == 0:
                    cat = "extinct"
                elif run:
                    cat = "runaway"
                elif log >= LOG_HI:
                    cat = "log_ge_300MB"
                elif log <= LOG_LO:
                    cat = "no_growth"
                else:
                    cat = "success"
                c500 = x.mean_rep_size.to_numpy(float)[-N_FINAL:]
                curves[(rule, D, P, seed)] = c500
                recs.append(dict(rule=rule, D=D, P=P, seed=seed, n_rows=len(x), log_MB=log / 1e6,
                                 n_active_final=act, max_rep_size_max=maxrep, sayama_ok=ok, runaway=run,
                                 used=ok and not run, category=cat, n_nan_final500=int(np.isnan(c500).sum()),
                                 B_run=float(np.nanmean(c500)) if np.isfinite(c500).any() else np.nan,
                                 C_run=float(x.cum_pattern_types.iloc[-1])))
    return pd.DataFrame(recs), curves


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def holm(p):
    p = np.asarray(p, float)
    adj = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    order = np.argsort(q)
    out = np.empty_like(q)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(q) - rank) * q[i])
        out[i] = min(1.0, running)
    adj[ok] = out
    return adj


def cliffs_delta(x, y):
    """delta = P(X>Y) - P(X<Y), from the Mann-Whitney U of x (ties count 1/2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 2 * res.statistic / (len(x) * len(y)) - 1, res.pvalue


def condition_table(runs, curves, n_boot, rng):
    """Per-condition A, B, C, A*B*C with Wilson / bootstrap 95% intervals; bootstrap replicates kept."""
    rows, boots = [], {}
    for (rule, D, P), g in runs.groupby(["rule", "D", "P"], sort=True):
        n = len(g)
        used = g.used.to_numpy()
        A = int(used.sum())
        lo, hi = wilson(A, n)
        b, c = g.B_run.to_numpy(), g.C_run.to_numpy()
        if A:
            ens = np.nanmean(np.nanmean(np.stack([curves[(rule, D, P, s)] for s in g.seed[used]]), axis=0))
            Bm, Cm = float(ens), float(c[used].mean())
            # conditional bootstrap over the successful runs (B, C)
            ok_b, ok_c = b[used], c[used]
            j = rng.integers(0, A, (n_boot, A))
            Bb, Cb = np.nanmean(ok_b[j], axis=1), ok_c[j].mean(axis=1)
            logC_b = np.log(ok_c[j]).mean(axis=1)
        else:
            Bm = Cm = np.nan
            Bb = Cb = logC_b = np.full(n_boot, np.nan)
        # unconditional bootstrap over all runs of the condition (A and A*B*C)
        idx = rng.integers(0, n, (n_boot, n))
        u = used[idx]
        Ab = u.sum(axis=1).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            Bu = np.where(u, np.nan_to_num(b[idx]), 0).sum(1) / Ab
            Cu = np.where(u, c[idx], 0).sum(1) / Ab
        ABCb = Ab * Bu * Cu
        boots[(rule, D, P)] = dict(A=Ab, B=Bb, C=Cb, logC=logC_b, ABC=ABCb)
        q = lambda v: np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).any() else (np.nan, np.nan)
        fc = g.category.value_counts()
        rows.append(dict(
            rule=rule, D=D, P=P, n_runs=n, A=A, A_wilson_lo=n * lo, A_wilson_hi=n * hi,
            B=Bm, B_lo=q(Bb)[0], B_hi=q(Bb)[1], B_mean_of_runs=float(np.nanmean(b[used])) if A else np.nan,
            B_median=float(np.nanmedian(b[used])) if A else np.nan,
            C=Cm, C_lo=q(Cb)[0], C_hi=q(Cb)[1], C_median=float(np.median(c[used])) if A else np.nan,
            C_geomean=float(np.exp(np.log(c[used]).mean())) if A else np.nan,
            ABC=A * Bm * Cm, ABC_lo=q(ABCb)[0], ABC_hi=q(ABCb)[1],
            **{f"n_{k}": int(fc.get(k, 0)) for k in CATS}))
    tab = pd.DataFrame(rows)
    for rule, g in tab.groupby("rule"):
        i = g.index
        tab.loc[i, "ABC_rel"] = g.ABC / g.ABC.max()
        tab.loc[i, "rank_ABC"] = g.ABC.rank(ascending=False, method="min")
        # alternative composite: mean percentile rank of A, B, C (equal weights, scale free)
        tab.loc[i, "composite_rank"] = (g.A.rank(pct=True) + g.B.rank(pct=True) + g.C.rank(pct=True)) / 3
        tab.loc[i, "ABgmC"] = g.A * g.B * g.C_geomean
        tab.loc[i, "rank_ABgmC"] = tab.loc[i, "ABgmC"].rank(ascending=False, method="min")
    return tab, boots


def pooled(tab, boots, rule, by, val, measure):
    """Mean over the other parameter of per-condition values at by=val, with a 95% interval: for A a Wilson
    interval on the pooled counts (scaled to runs per condition; a bootstrap gives the zero-width interval
    [30, 30] when every run succeeds), for B, C and A x B x C the percentile bootstrap."""
    other = "D" if by == "P" else "P"
    g = tab[(tab.rule == rule) & (tab[by] == val)]
    if g.empty:
        return np.nan, np.nan, np.nan, 0
    mean = float(np.nanmean(g[measure]))
    if measure == "A":
        k, n = int(g.A.sum()), int(g.n_runs.sum())
        lo, hi = wilson(k, n)
        per = n / len(g)  # mean runs per condition, so that k / n * per = mean A
        return mean, lo * per, hi * per, len(g[other].unique())
    keys = [(rule, int(r.D), float(r.P)) for r in g.itertuples()]
    reps = np.nanmean(np.array([boots[k][measure] for k in keys]), axis=0)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return mean, lo, hi, len(g[other].unique())


# =============================== statistics ===============================
def ols_full(X, y):
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    r = y - X @ beta
    s2 = r @ r / (n - k)
    V = XtXi * s2
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    V3 = XtXi @ (X.T * (r / (1 - h)) ** 2) @ X @ XtXi
    r2 = 1 - (r @ r) / np.sum((y - y.mean()) ** 2)
    return beta, V, V3, r2, n - k


def wald_F(beta, V, idx, dof):
    R = np.zeros((len(idx), len(beta)))
    R[np.arange(len(idx)), idx] = 1
    rb = R @ beta
    F = float(rb @ np.linalg.solve(R @ V @ R.T, rb) / len(idx))
    return F, float(stats.f.sf(F, len(idx), dof))


def logit_fit(X, y, iters=200):
    """Logistic regression by Newton-Raphson on standardized columns; returns beta, cov, loglik."""
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0)
    Z = np.column_stack([X[:, 0], (X[:, 1:] - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + 1e-10 * np.eye(len(w))
        step = np.linalg.solve(H, Z.T @ (y - p))
        w += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-Z @ w)), 1e-15, 1 - 1e-15)
    H = Z.T @ (Z * (p * (1 - p))[:, None])
    T = np.eye(len(w))
    T[0, 1:] = -mu / sd
    T[1:, 1:] = np.diag(1 / sd)
    beta = T @ w
    cov = T @ np.linalg.inv(H) @ T.T
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return beta, cov, ll


def loglik_null(y):
    m = np.clip(y.mean(), 1e-15, 1 - 1e-15)
    return float(np.sum(y * np.log(m) + (1 - y) * np.log(1 - m)))


def regression_data(runs, rule):
    g = runs[runs.rule == rule]
    cond = g.groupby(["D", "P"]).used.sum().reset_index().rename(columns={"used": "value"})
    ok = g[g.used]
    return {"A": cond.assign(value=cond.value.astype(float)),
            "B": ok[["D", "P"]].assign(value=ok.B_run.to_numpy()),
            "C": ok[["D", "P"]].assign(value=ok.C_run.to_numpy()),
            "logC": ok[["D", "P"]].assign(value=np.log(ok.C_run.to_numpy())),
            "A_bin": g[["D", "P"]].assign(value=g.used.astype(float).to_numpy())}


def mathematica_data():
    m = rsd.load_mathematica()
    A = m["A"].copy()
    rows = []
    for r in A.itertuples():
        rows += [(r.D, r.P, 1.0)] * int(r.value) + [(r.D, r.P, 0.0)] * int(30 - r.value)
    return {"A": A, "B": m["B"], "C": m["C"], "logC": m["C"].assign(value=np.log(m["C"].value)),
            "A_bin": pd.DataFrame(rows, columns=["D", "P", "value"])}


def run_regressions(datasets):
    rows, joint = [], []
    blocks = {"D terms (D, D^2, DP)": [1, 2, 5], "P terms (P, P^2, DP)": [3, 4, 5], "DP": [5]}
    for name, data in datasets.items():
        for m in ("A", "B", "C", "logC"):
            df = data[m]
            X = rsd.design(df.D, df.P)
            y = df.value.to_numpy(float)
            beta, V, V3, r2, dof = ols_full(X, y)
            # cross-check with scripts/regress_spatial_dyadic.ols
            b_chk, se_chk, *_ = rsd.ols(X, y)
            assert np.allclose(b_chk, beta) and np.allclose(se_chk, np.sqrt(np.diag(V)))
            se, se3 = np.sqrt(np.diag(V)), np.sqrt(np.diag(V3))
            for t, b, s, s3 in zip(TERMS, beta, se, se3):
                rows.append(dict(data=name, response=m, model="OLS", n=len(y), term=t, coef=b, se=s,
                                 p=2 * stats.t.sf(abs(b / s), dof), se_hc3=s3,
                                 p_hc3=2 * stats.t.sf(abs(b / s3), dof), r2=r2))
            for bname, idx in blocks.items():
                F, p = wald_F(beta, V, idx, dof)
                F3, p3 = wald_F(beta, V3, idx, dof)
                joint.append(dict(data=name, response=m, model="OLS", block=bname, stat=F, p=p,
                                  stat_hc3=F3, p_hc3=p3))
        df = data["A_bin"]
        X = rsd.design(df.D, df.P)
        y = df.value.to_numpy(float)
        beta, cov, ll = logit_fit(X, y)
        ll0 = loglik_null(y)
        se = np.sqrt(np.diag(cov))
        for t, b, s in zip(TERMS, beta, se):
            rows.append(dict(data=name, response="A", model="logit", n=len(y), term=t, coef=b, se=s,
                             p=2 * stats.norm.sf(abs(b / s)), se_hc3=np.nan, p_hc3=np.nan,
                             r2=1 - ll / ll0))  # McFadden pseudo-R^2
        for bname, idx in blocks.items():
            keep = [i for i in range(6) if i not in idx]
            _, _, llr = logit_fit(X[:, keep], y)
            LR = 2 * (ll - llr)
            joint.append(dict(data=name, response="A", model="logit (LR test)", block=bname, stat=LR,
                              p=float(stats.chi2.sf(LR, len(idx))), stat_hc3=np.nan, p_hc3=np.nan))
    return pd.DataFrame(rows), pd.DataFrame(joint)


def cochran_armitage(k, n, x):
    k, n, x = map(lambda a: np.asarray(a, float), (k, n, x))
    N, p = n.sum(), k.sum() / n.sum()
    T = np.sum(x * (k - n * p))
    var = p * (1 - p) * (np.sum(n * x ** 2) - np.sum(n * x) ** 2 / N)
    z = T / math.sqrt(var) if var > 0 else np.nan
    return z, 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan


def trend_tests(runs, rule):
    """Monotone trend in P pooled over D (and in D pooled over P) for A (per-run 0/1), B, C, log C.
    Returns (one row per trend and measure, one row per stratum of the per-stratum Spearman)."""
    g = runs[runs.rule == rule]
    ok = g[g.used]
    out, strata_rows = [], []
    series = {"A": (g, g.used.astype(float)), "B": (ok, ok.B_run), "C": (ok, ok.C_run),
              "logC": (ok, np.log(ok.C_run))}
    for var, strat in (("P", "D"), ("D", "P")):
        for m, (d, y) in series.items():
            y = np.asarray(y, float)
            rho, prho = stats.spearmanr(d[var], y)
            # linear slope adjusting for the other parameter as a factor (HC3 SE)
            levels = sorted(d[strat].unique())
            X = np.column_stack([d[var].to_numpy(float)] + [(d[strat] == lv).to_numpy(float) for lv in levels])
            beta, V, V3, _, dof = ols_full(X, y)
            slope, se3 = beta[0], math.sqrt(V3[0, 0])
            # per-stratum Spearman, combined by Stouffer on Fisher-z
            zs, pos, neg, rhos = [], 0, 0, []
            for lv in levels:
                s = d[strat] == lv
                varies = bool(s.sum() > 3 and np.nanstd(y[s.to_numpy()]) > 0)
                r, pr = stats.spearmanr(d[var][s], y[s.to_numpy()]) if varies else (np.nan, np.nan)
                strata_rows.append(dict(rule=rule, trend_in=var, pooled_over=strat, measure=m, level=lv,
                                        n=int(s.sum()), rho=r, p=pr, varies=varies))
                if varies:
                    zs.append(np.arctanh(np.clip(r, -0.999999, 0.999999)) * math.sqrt(s.sum() - 3))
                    rhos.append(f"{strat}={lv:g}:{r:+.2f}{'*' if pr < 0.05 else ''}")
                    pos += int(r > 0 and pr < 0.05)
                    neg += int(r < 0 and pr < 0.05)
            Z = float(np.sum(zs) / math.sqrt(len(zs))) if zs else np.nan
            row = dict(rule=rule, trend_in=var, pooled_over=strat, measure=m, n=len(y), spearman_rho=rho,
                       spearman_p=prho, slope_adj=slope, slope_se_hc3=se3,
                       slope_p_hc3=2 * stats.t.sf(abs(slope / se3), dof), n_strata=len(zs),
                       strata_pos_sig=pos, strata_neg_sig=neg, strata_rho=" ".join(rhos), stouffer_Z=Z,
                       stouffer_p=2 * stats.norm.sf(abs(Z)) if np.isfinite(Z) else np.nan)
            if m == "A":
                tot = g.groupby(var).used.agg(["sum", "count"])
                row["cochran_armitage_z"], row["cochran_armitage_p"] = cochran_armitage(
                    tot["sum"], tot["count"], tot.index.to_numpy(float))
            out.append(row)
    return pd.DataFrame(out), pd.DataFrame(strata_rows)


def compare(gx, gy, label_x, label_y):
    """A: Fisher exact + risk difference (= Cliff's delta for 0/1); B, C: Mann-Whitney + Cliff's delta."""
    rows = []
    kx, nx, ky, ny = int(gx.used.sum()), len(gx), int(gy.used.sum()), len(gy)
    if nx and ny:
        p = stats.fisher_exact([[kx, nx - kx], [ky, ny - ky]])[1]
        rows.append(dict(x=label_x, y=label_y, measure="A", n_x=nx, n_y=ny, stat_x=kx / nx, stat_y=ky / ny,
                         test="Fisher exact (success fraction)", p=p, cliffs_delta=kx / nx - ky / ny))
    for m, col in (("B", "B_run"), ("C", "C_run")):
        x, y = gx[gx.used][col].dropna().to_numpy(), gy[gy.used][col].dropna().to_numpy()
        if len(x) and len(y):
            dlt, p = cliffs_delta(x, y)
            rows.append(dict(x=label_x, y=label_y, measure=m, n_x=len(x), n_y=len(y), stat_x=np.median(x),
                             stat_y=np.median(y), test="Mann-Whitney U (medians shown)", p=p, cliffs_delta=dlt))
    return rows


def contrasts(runs, tab, rule):
    g = runs[runs.rule == rule]
    t = tab[tab.rule == rule]
    cond = lambda D, P: g[(g.D == D) & (g.P == P)]
    best = t.sort_values("ABC", ascending=False).iloc[0]
    bD, bP = int(best.D), float(best.P)
    c_best = pd.DataFrame(compare(cond(bD, bP), cond(*ORIGINAL), f"best ({bD},{bP})", "original (100,0)"))
    c_best["p_holm"] = holm(c_best.p)
    # each of the 59 other conditions vs the original, Holm within measure
    rows = []
    for D in D_VALUES:
        for P in P_VALUES:
            if (D, P) != ORIGINAL and len(cond(D, P)):
                for r in compare(cond(D, P), cond(*ORIGINAL), f"({D},{P})", "original (100,0)"):
                    rows.append(dict(D=D, P=P, **r))
    c_all = pd.DataFrame(rows)
    if len(c_all):
        c_all["p_holm"] = np.nan
        for m, gg in c_all.groupby("measure"):
            c_all.loc[gg.index, "p_holm"] = holm(gg.p)
    # P=0 vs each P>0 pooled over D; D=100 vs each D<100 pooled over P (Holm across each table)
    cP = pd.DataFrame([r for P in P_VALUES[1:] if len(g[g.P == P])
                       for r in compare(g[g.P == P], g[g.P == 0.0], f"P={P}", "P=0")])
    cD = pd.DataFrame([r for D in D_VALUES[:-1] if len(g[g.D == D])
                       for r in compare(g[g.D == D], g[g.D == 100], f"D={D}", "D=100")])
    for c in (cP, cD):
        if len(c):
            c["p_holm"] = holm(c.p)
    return c_best, c_all, cP, cD, (bD, bP)


# =============================== figures ===============================
def fmt_val(v, kind):
    if not np.isfinite(v):
        return ""
    return {"A": f"{v:.0f}", "B": f"{v:.2f}", "C": f"{v / 1e3:.1f}", "ABC": f"{v / 1e6:.2f}"}[kind]


HEAT_SPECS = [  # key, file name, colourbar label, title
    ("A", "P-D-final-num-runs", "Successful runs (of 30)", "(a) Successful runs A"),
    ("B", "P-D-final-size", "Mean active cells per\nreplication event", "(b) Replication-event size B, t = 1500-2000"),
    ("C", "P-D-final-nctypes", "Cumulative pattern\ntypes (x10$^3$)", "(c) Pattern types C, t = 2000 (end of run)"),
    ("ABC", "P-D-overall-performance", "A x B x C (x10$^6$)", "(d) Overall performance A x B x C"),
]


def heat_panel(fig, ax, cax, t, key, label, title, best=None):
    grid = np.full((len(P_VALUES), len(D_VALUES)), np.nan)
    for r in t.itertuples():
        grid[P_VALUES.index(float(r.P)), D_VALUES.index(int(r.D))] = getattr(r, key)
    scale = {"C": 1e3, "ABC": 1e6}.get(key, 1.0)
    shown = grid / scale
    finite = shown[np.isfinite(shown)]
    vmin, vmax = (finite.min(), finite.max()) if finite.size else (0, 1)
    # rows: P = 0 at the TOP ... 1 at the bottom; columns: D = 10 ... 100 left to right,
    # so (D, P) = (L, 0) is the top-right cell, as in the original Mathematica MatrixPlot
    cmap = sns.color_palette(HEAT_CMAP, as_cmap=True)
    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="upper",
                   interpolation="nearest")
    norm = Normalize(vmin, vmax)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                rgb = cmap(norm(shown[i, j]))[:3]
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(j, i, fmt_val(grid[i, j], key), ha="center", va="center", fontsize=5.6,
                        color="white" if lum < 0.45 else "#1a1a1a")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5, color="#888888")
    ax.set_xticks(range(len(D_VALUES)), D_VALUES)
    ax.set_yticks(range(len(P_VALUES)), [f"{p:.1f}" for p in P_VALUES])
    ax.set_xlabel("Spatial interaction range $D$")
    ax.set_ylabel("Dyadic interaction probability $P$")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    oi, oj = P_VALUES.index(ORIGINAL[1]), D_VALUES.index(ORIGINAL[0])
    # each box is a black line over a wider white one, so it stays visible on the darkest cells too
    ax.add_patch(Rectangle((oj - 0.47, oi - 0.47), 0.94, 0.94, fill=False, ec="white", lw=2.9, zorder=5))
    ax.add_patch(Rectangle((oj - 0.47, oi - 0.47), 0.94, 0.94, fill=False, ec="black", lw=1.3, zorder=6))
    if best is not None:
        bi, bj = P_VALUES.index(best[1]), D_VALUES.index(best[0])
        ax.add_patch(Rectangle((bj - 0.43, bi - 0.43), 0.86, 0.86, fill=False, ec="white", lw=2.4, zorder=5))
        ax.add_patch(Rectangle((bj - 0.43, bi - 0.43), 0.86, 0.86, fill=False, ec="black", lw=1.1,
                               ls=(0, (2, 1.4)), zorder=6))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.5)
    ax.set_title(title, fontsize=8, loc="left", pad=4)


def fig5(tab, out, best):
    t = tab[tab.rule == "intended"]
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.64))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 0.035, 0.42, 1, 0.035], wspace=0.05, hspace=0.62,
                          left=0.08, right=0.92, top=0.94, bottom=0.12)
    for n, (key, _, label, title) in enumerate(HEAT_SPECS):
        r, c = divmod(n, 2)
        heat_panel(fig, fig.add_subplot(gs[r, 3 * c]), fig.add_subplot(gs[r, 3 * c + 1]), t, key, label, title,
                   best=best if key == "ABC" else None)
    fig.text(0.5, 0.0, "Solid box: original setting (D, P) = (L, 0) (top-right cell). Dashed box in (d): "
             "condition with the largest A x B x C. B: final 500 steps; C: after all 2000 steps.",
             ha="center", fontsize=6.5, color="#444444")
    savefig(fig, out, "fig5_heatmaps")
    for key, fname, label, title in HEAT_SPECS:  # single panels, drop-in file names
        fig = plt.figure(figsize=(92 * MM, 58 * MM))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.04], wspace=0.06, left=0.15, right=0.86, bottom=0.2, top=0.9)
        heat_panel(fig, fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), t, key, label,
                   title[4:], best=best if key == "ABC" else None)
        savefig(fig, out, fname)


def label_set(best):
    """Labelled points: the original setting, the best condition and the Fig. 7 conditions."""
    keys = {ORIGINAL: "original (L, 0)"}
    for D, P, _ in SNAP_CONDS:
        keys.setdefault((D, P), f"({D}, {P:.1f})")
    keys[best] = keys.get(best, f"({best[0]}, {best[1]:.1f})") + " best"
    return keys


# label offsets (points) per panel of the 2-D version: (A,B), (A,C), (B,C); default (4, 3)
PAIR_OFFSETS = [
    {ORIGINAL: (5, -2), (10, 0.0): (-48, 4), (20, 0.2): (-50, -4), (100, 1.0): (-62, -16)},
    {ORIGINAL: (-10, 7), (10, 0.0): (-48, -10), (20, 0.2): (-52, 6), (100, 1.0): (-66, 2)},
    {ORIGINAL: (6, -12), (10, 0.0): (-6, -14), (20, 0.2): (-12, 18), (100, 1.0): (6, -3)},
]


def fig6(tab, out, best):
    t = tab[tab.rule == "intended"].copy()
    cmap = sns.color_palette(HEAT_CMAP, as_cmap=True)
    norm = Normalize(0, 1)
    labels = label_set(best)
    lab_mask = np.array([(int(r.D), float(r.P)) in labels for r in t.itertuples()])
    # ---- 3-D
    fig = plt.figure(figsize=(130 * MM, 108 * MM))
    ax = fig.add_axes([0.0, 0.02, 0.80, 0.96], projection="3d")
    z0 = 0.0
    for r in t.itertuples():  # stems to the floor help to read depth
        ax.plot([r.A, r.A], [r.B, r.B], [z0, r.C / 1e3], color="#b5b5b5", lw=0.4, zorder=1)
    ax.scatter(t.A, t.B, t.C / 1e3, c=t.ABC_rel, cmap=cmap, norm=norm, s=np.where(lab_mask, 30, 18),
               edgecolors=np.where(lab_mask, "black", "#444444"), linewidths=np.where(lab_mask, 0.9, 0.3),
               depthshade=False, zorder=3)
    # labels: annotate the projected 2-D position of each labelled point (leader lines)
    # (10, 0) and (20, 0.2) project next to each other: labels go to opposite sides, leader lines end at
    # the marker centre (shrinkB=0) so each label points at exactly one marker
    off3 = {ORIGINAL: (-10, 22), (10, 0.0): (12, 38), (20, 0.2): (-40, -40), (100, 1.0): (-30, 18)}
    # all limits and the view are fixed BEFORE the projection is read: a later set_zlim would move the
    # points away from the label anchors
    ax.set_xlabel("A: successful runs (of 30)", fontsize=7, labelpad=2)
    ax.set_ylabel("B: replication-event size", fontsize=7, labelpad=2)
    ax.set_zlabel("C: pattern types (x10$^3$)", fontsize=7, labelpad=2)
    ax.set_zlim(z0, None)
    ax.tick_params(labelsize=6, pad=0)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.975, 0.975, 0.975, 1.0))
        a.pane.set_edgecolor("#cccccc")
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect(None, zoom=0.86)
    fig.canvas.draw()
    for (D, P), lab in labels.items():
        r = t[(t.D == D) & (t.P == P)]
        if len(r):
            r = r.iloc[0]
            x2, y2, _ = proj3d.proj_transform(r.A, r.B, r.C / 1e3, ax.get_proj())
            ax.annotate(lab, xy=(x2, y2), xytext=off3.get((D, P), (10, 10)), textcoords="offset points",
                        fontsize=6, ha="center", va="center", color="#1a1a1a", zorder=10,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="#222222", shrinkA=1, shrinkB=0))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = fig.add_axes([0.84, 0.25, 0.025, 0.5])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Relative overall performance\nA x B x C / max (as Fig. 5d)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    savefig(fig, out, "fig6_3d")
    # ---- 2-D pair panels
    fig = plt.figure(figsize=(DOUBLE_COL, 62 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.04], wspace=0.42, left=0.07, right=0.93, bottom=0.18,
                          top=0.95)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    pairs = [("A", "B", "A: successful runs (of 30)", "B: replication-event size"),
             ("A", "C", "A: successful runs (of 30)", "C: pattern types (x10$^3$)"),
             ("B", "C", "B: replication-event size", "C: pattern types (x10$^3$)")]
    jit = np.random.default_rng(1).uniform(-A_JITTER, A_JITTER, len(t))
    for ax, (xk, yk, xl, yl), offs in zip(axes, pairs, PAIR_OFFSETS):
        xv = t[xk].to_numpy(float) + (jit if xk == "A" else 0)
        yv = t[yk].to_numpy(float) / (1e3 if yk == "C" else 1)
        ax.scatter(xv[~lab_mask], yv[~lab_mask], c=t.ABC_rel[~lab_mask], cmap=cmap, norm=norm, s=15,
                   edgecolors="#444444", linewidths=0.3, zorder=3)
        ax.scatter(xv[lab_mask], yv[lab_mask], c=t.ABC_rel[lab_mask], cmap=cmap, norm=norm, s=24,
                   edgecolors="black", linewidths=0.9, zorder=4)
        for (D, P), lab in labels.items():
            m = ((t.D == D) & (t.P == P)).to_numpy()
            if m.any():
                ax.annotate(lab, (xv[m][0], yv[m][0]), xytext=offs.get((D, P), (4, 3)), textcoords="offset points",
                            fontsize=5.6, color="#1a1a1a", zorder=5,
                            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
                            arrowprops=dict(arrowstyle="-", lw=0.4, color="#555555", shrinkA=0, shrinkB=2.5))
        ax.set_xlabel(xl)
        ax.set_ylabel(yl, labelpad=2)
        if xk == "B":
            ax.set_xlim(None, xv.max() + 0.45 * (xv.max() - xv.min()))
        sns.despine(ax=ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=fig.add_subplot(gs[0, 3]))
    cb.set_label("A x B x C / max", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.text(0.5, -0.02, f"A is an integer count; points are jittered horizontally by up to +/-{A_JITTER} so that "
             "conditions with equal A do not overlap.", ha="center", fontsize=6.5, color="#444444")
    savefig(fig, out, "fig6_pairs")


def fig_trends(tab, boots, out):
    t = tab[tab.rule == "intended"]
    meas = [("A", "A: successful runs (of 30)"), ("B", "B: replication-event size"),
            ("C", "C: pattern types (x10$^3$)")]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 110 * MM), gridspec_kw=dict(wspace=0.38, hspace=0.5))
    for row, (by, other, vals, ovals, pal) in enumerate([("P", "D", P_VALUES, D_VALUES, "flare"),
                                                         ("D", "P", D_VALUES, P_VALUES, "crest")]):
        colors = sns.color_palette(pal, len(ovals))
        for ax, (m, lab) in zip(axes[row], meas):
            sc = 1e3 if m == "C" else 1
            for o, col in zip(ovals, colors):
                g = t[t[other] == o].sort_values(by)
                ax.plot(g[by], g[m] / sc, color=col, lw=0.7, alpha=0.75, marker="o", ms=1.8)
            pm = [pooled(tab, boots, "intended", by, v, m) for v in vals]
            ax.fill_between(vals, [p[1] / sc for p in pm], [p[2] / sc for p in pm], color="#222222", alpha=0.18,
                            lw=0, zorder=4)
            ax.plot(vals, [p[0] / sc for p in pm], color="#111111", lw=1.6, zorder=5)
            ax.set_xlabel({"P": "Dyadic interaction probability $P$", "D": "Spatial interaction range $D$"}[by])
            ax.set_ylabel(lab)
            ax.set_xticks(vals)
            if by == "D":
                ax.set_xticklabels([str(v) if v in (10, 30, 50, 70, 100) else "" for v in vals])
            sns.despine(ax=ax)
        handles = [Line2D([], [], color=c, lw=1, label=f"{other}={o if other == 'D' else f'{o:.1f}'}")
                   for o, c in zip(ovals, colors)]
        handles.append(Line2D([], [], color="#111111", lw=1.6, label=f"mean over {other} (95% CI)"))
        axes[row, 2].legend(handles=handles, fontsize=5.3, frameon=False, ncol=1, loc="center left",
                            bbox_to_anchor=(1.0, 0.5), handlelength=1.2)
    savefig(fig, out, "figS_trends")


def fig_failures(runs, out):
    """Stacked failure counts; each column (pooled by P / by D) shares one y-axis across the two rules so that
    intended and as-coded can be compared directly."""
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 100 * MM), sharey="col",
                             gridspec_kw=dict(wspace=0.22, hspace=0.62))
    cats = ["extinct", "no_growth", "runaway", "log_ge_300MB"]
    tops = {"P": 1.0, "D": 1.0}
    for row, rule in enumerate(["intended", "as_coded"]):
        g = runs[runs.rule == rule]
        for col, (by, vals) in enumerate([("P", P_VALUES), ("D", D_VALUES)]):
            ax = axes[row, col]
            bottom = np.zeros(len(vals))
            for c in cats:
                h = np.array([((g[by] == v) & (g.category == c)).sum() for v in vals], float)
                ax.bar(range(len(vals)), h, bottom=bottom, color=FAIL_COLORS[c], width=0.7, label=CAT_LABEL[c],
                       edgecolor="white", lw=0.3)
                bottom += h
            n = np.array([(g[by] == v).sum() for v in vals])
            same_n = len(set(n.tolist())) == 1
            # the count of failed runs above each bar; the denominator goes into the axis label when it is the
            # same for all bars (otherwise "k/n", which is wider)
            for i, (b, nn) in enumerate(zip(bottom, n)):
                ax.text(i, b + 0.4, f"{int(b)}" if same_n else f"{int(b)}/{nn}", ha="center", va="bottom",
                        fontsize=5.5 if same_n else 4.5)
            ax.set_xticks(range(len(vals)), [f"{v:.1f}" if by == "P" else str(v) for v in vals])
            other = "D" if by == "P" else "P"
            ax.set_xlabel(f"{by} (pooled over {other}; " + (f"{n[0]} runs each)" if same_n else "runs per bar vary)"))
            ax.set_ylabel("Failed runs")
            ax.set_title(f"{'Intended' if rule == 'intended' else 'As-coded'} dyadic rule", fontsize=7.5, loc="left")
            tops[by] = max(tops[by], bottom.max())
            sns.despine(ax=ax)
    for col, by in enumerate(("P", "D")):
        axes[0, col].set_ylim(0, tops[by] * 1.15 + 1)  # sharey="col": applies to both rows
    handles = [Patch(color=FAIL_COLORS[c], label=CAT_LABEL[c]) for c in cats]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.03))
    savefig(fig, out, "figS_failures")


# =============================== snapshots ===============================
def render_hue(cfg):
    """Sayama's MatrixPlot colouring: cell type c -> Hue[c/k] (full saturation and brightness); empty = white."""
    img = np.ones(cfg.shape + (3,), np.float32)
    occ = cfg > 0
    if occ.any():
        h = (cfg[occ].astype(float) / K_TYPES) % 1.0
        img[occ] = hsv_to_rgb(np.stack([h, np.ones_like(h), np.ones_like(h)], -1))
    return img


def snapshot_stats(cfg):
    occ = cfg > 0
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    pats, sizes = {}, []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        sub = np.where(lab[sl] == i, cfg[sl], 0)
        key = (sub.shape, sub.astype(np.int16).tobytes())
        pats.setdefault(key, []).append(int((sub > 0).sum()))
        sizes.append(int((sub > 0).sum()))
    n_act = int(occ.sum())
    dom = max(pats.values(), key=lambda v: sum(v)) if pats else []
    types, counts = np.unique(cfg[occ], return_counts=True)
    q = counts / counts.sum() if counts.size else counts
    return dict(n_active=n_act, n_components=int(n), n_cell_types=int(types.size),
                n_patterns=len(pats), largest_component=max(sizes) if sizes else 0,
                mean_component=float(np.mean(sizes)) if sizes else 0.0,
                dominant_pattern_share=float(sum(dom) / n_act) if n_act else np.nan,
                dominant_pattern_copies=len(dom),
                eff_n_types=float(1 / np.sum(q ** 2)) if q.size else np.nan)


def fmt_C(c):
    """C as an integer below 1000, else in thousands (e.g. 39, 601, 2.4k)."""
    return "n/a" if not np.isfinite(c) else (f"{c:.0f}" if c < 1000 else f"{c / 1e3:.1f}k")


def snapshot_exclusion(r):
    """Why a seed is not a Fig. 7 candidate at its snapshot time ('' if it is one)."""
    if r.candidate:
        return ""
    if not (r.run_used == True):  # noqa: E712 (bool, numpy bool or None)
        return {"no_growth": "no growth", "log_ge_300MB": "log >= 300 MB"}.get(r.run_category, r.run_category) \
            or "no run record"
    return f"{int(r.n_active)} cells (< {SNAP_MIN_ACTIVE})"


def snapshots(runs, out):
    """Fig. 7 seed choice. Candidates (per condition and snapshot time): seeds 0-7 whose run is successful
    (Sayama's criterion, not runaway) and has >= SNAP_MIN_ACTIVE active cells at that time, i.e. a grown
    population (successful runs with only 26-126 cells exist; they are not representative of a condition).
    At the latest time in (2000, 1750, 1500) with a candidate, take the candidate whose C is closest to the
    median C of all successful runs of the condition (ties: lower seed). The dominant-pattern share is
    summarised over the same candidates (median, range, cell-weighted), for every snapshot time."""
    g = runs[runs.rule == "intended"]
    stat_rows, sel_rows = [], []
    for D, P, lab in SNAP_CONDS:
        d = SNAP_ROOT / "intended" / f"D{D}_P{P:.1f}"
        csvdir = SD_ROOT / "intended" / f"D{D}_P{P:.1f}"
        for s in SNAP_SEEDS:
            r = g[(g.D == D) & (g.P == P) & (g.seed == s)]
            for t in SNAP_TIMES:
                f = d / f"seed_{s}_t{t}.npy"
                if not f.exists():
                    continue
                cfg = np.load(f)
                st = snapshot_stats(cfg)
                x = pd.read_csv(csvdir / f"seed_{s}.csv", usecols=["step", "n_active", "mean_rep_size"])
                ref_t = x.loc[x.step == t, "n_active"]
                recent = x.loc[(x.step >= t - 50) & (x.step < t), "mean_rep_size"].to_numpy(float)
                ref_t1 = x.loc[x.step == t - 1, "n_active"]
                used = bool(r.used.iloc[0]) if len(r) else None
                stat_rows.append(dict(D=D, P=P, seed=s, t=t, **st,
                                      csv_n_active_step_t=int(ref_t.iloc[0]) if len(ref_t) else np.nan,
                                      csv_n_active_step_t_minus_1=int(ref_t1.iloc[0]) if len(ref_t1) else np.nan,
                                      recent_rep_size=float(np.nanmean(recent)) if np.isfinite(recent).any() else np.nan,
                                      run_used=used, run_category=r.category.iloc[0] if len(r) else "",
                                      C_run=float(r.C_run.iloc[0]) if len(r) else np.nan,
                                      candidate=bool(used is True and st["n_active"] >= SNAP_MIN_ACTIVE)))
    st = pd.DataFrame(stat_rows)
    if len(st):
        st["excluded_because"] = [snapshot_exclusion(r) for r in st.itertuples()]
        st.to_csv(out / "snapshot_stats.csv", index=False)
    for f in out.glob("snap_D*_P*_seed*_t*.png"):  # this script's own single panels; the selection may change
        f.unlink()

    # typical dominance per condition and time, over the candidates (the population the panel is drawn from);
    # the median over all seeds with any active cell is kept only to show why it is not used (tiny or failed
    # populations have a share near 1 almost by definition)
    dom_rows = []
    for D, P, lab in SNAP_CONDS:
        for t in SNAP_TIMES:
            c = st[(st.D == D) & (st.P == P) & (st.t == t)] if len(st) else st
            if not len(c):
                continue
            cc = c[c.candidate]
            share = cc.dominant_pattern_share.to_numpy(float)
            dom_rows.append(dict(
                D=D, P=P, label=lab, t=t, n_seeds=len(c), n_candidates=len(cc),
                candidate_seeds=" ".join(str(int(v)) for v in cc.seed),
                dom_median=float(np.median(share)) if len(cc) else np.nan,
                dom_min=float(share.min()) if len(cc) else np.nan,
                dom_max=float(share.max()) if len(cc) else np.nan,
                dom_cell_weighted=float((share * cc.n_active).sum() / cc.n_active.sum()) if len(cc) else np.nan,
                n_dom_gt_half=int((share > 0.5).sum()),
                patterns_median=float(cc.n_patterns.median()) if len(cc) else np.nan,
                active_median=float(cc.n_active.median()) if len(cc) else np.nan,
                dom_median_all_nonempty_seeds=float(np.nanmedian(c[c.n_active > 0].dominant_pattern_share))
                if (c.n_active > 0).any() else np.nan))
    dom = pd.DataFrame(dom_rows)
    dom.to_csv(out / "snapshot_dominance_by_time.csv", index=False)

    chosen = {}
    for D, P, lab in SNAP_CONDS:
        cand = st[(st.D == D) & (st.P == P)] if len(st) else st
        pick = None
        for min_act, rule_note in ((SNAP_MIN_ACTIVE, f"successful and >= {SNAP_MIN_ACTIVE} active cells"),
                                   (1, f"FALLBACK: no successful seed with >= {SNAP_MIN_ACTIVE} cells; "
                                       "successful and non-empty")):
            for t in SNAP_TIMES:
                c = cand[(cand.t == t) & (cand.run_used == True) & (cand.n_active >= min_act)] \
                    if len(cand) else cand  # noqa: E712
                if len(c):
                    medC = float(g[(g.D == D) & (g.P == P) & g.used].C_run.median())
                    c = c.assign(dist=(c.C_run - medC).abs()).sort_values(["dist", "seed"]).reset_index(drop=True)
                    pick = c.iloc[0].copy()
                    pick["cond_median_C"], pick["n_cand"], pick["rule_note"] = medC, len(c), rule_note
                    pick["cand_seeds"] = " ".join(str(int(v)) for v in sorted(c.seed))
                    sh = c.dominant_pattern_share.to_numpy(float)
                    pick["cand_dom_median"], pick["cand_dom_min"], pick["cand_dom_max"] = \
                        float(np.median(sh)), float(sh.min()), float(sh.max())
                    pick["cand_dom_cell_weighted"] = float((sh * c.n_active).sum() / c.n_active.sum())
                    pick["dom_rank"] = int((sh > pick.dominant_pattern_share).sum() + 1)  # 1 = most dominated
                    break
            if pick is not None:
                break
        if pick is None:
            sel_rows.append(dict(D=D, P=P, label=lab, seed=np.nan, t=np.nan, n_candidates=0,
                                 note="no snapshot of a successful non-extinct seed available yet"))
            continue
        chosen[(D, P)] = pick
        sel_rows.append(dict(
            D=D, P=P, label=lab, seed=int(pick.seed), t=int(pick.t), n_candidates=int(pick.n_cand),
            candidate_seeds=pick.cand_seeds, C_run=pick.C_run, cond_median_C=pick.cond_median_C,
            n_active=int(pick.n_active), n_patterns=int(pick.n_patterns),
            dominant_pattern_share=pick.dominant_pattern_share, cand_dom_median=pick.cand_dom_median,
            cand_dom_min=pick.cand_dom_min, cand_dom_max=pick.cand_dom_max,
            cand_dom_cell_weighted=pick.cand_dom_cell_weighted, dom_rank_among_candidates=int(pick.dom_rank),
            candidate_rule=pick.rule_note,
            note="seed whose C is closest to the condition median C (all successful runs)"))
        cfg = np.load(SNAP_ROOT / "intended" / f"D{D}_P{P:.1f}" / f"seed_{int(pick.seed)}_t{int(pick.t)}.npy")
        img = np.repeat(np.repeat(render_hue(cfg), 4, 0), 4, 1)
        plt.imsave(out / f"snap_D{D}_P{P:.1f}_seed{int(pick.seed)}_t{int(pick.t)}.png", img)
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(out / "snapshot_selection.csv", index=False)
    # combined Fig. 7 (missing panels drawn as placeholders)
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COL, 60 * MM), gridspec_kw=dict(wspace=0.08))
    for ax, (D, P, lab), pl in zip(axes, SNAP_CONDS, "abcd"):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6); sp.set_color("#888888")
        if (D, P) in chosen:
            pick = chosen[(D, P)]
            cfg = np.load(SNAP_ROOT / "intended" / f"D{D}_P{P:.1f}" / f"seed_{int(pick.seed)}_t{int(pick.t)}.npy")
            ax.imshow(render_hue(cfg), interpolation="nearest", origin="upper")
            ax.set_xlabel(f"seed {int(pick.seed)}, t = {int(pick.t)}\n{int(pick.n_active)} active cells\n"
                          f"{int(pick.n_patterns)} distinct patterns present\n"
                          f"(chosen from {int(pick.n_cand)} candidate runs)", fontsize=6)
        else:
            ax.imshow(np.ones((L, L, 3)), interpolation="nearest", origin="upper")
            ax.text(0.5, 0.5, "snapshot\npending", ha="center", va="center", transform=ax.transAxes,
                    fontsize=7, color="#888888")
        ax.set_title(f"({pl})  {lab}", fontsize=7, loc="left")
    savefig(fig, out, "fig7_snapshots")
    # all seeds, for transparency: chosen seed framed in black, non-candidates titled in grey with the reason
    if len(st):
        tt = {}
        for D, P, _ in SNAP_CONDS:
            c = st[(st.D == D) & (st.P == P)]
            tt[(D, P)] = int(c.t.max()) if len(c) else None
        fig, axes = plt.subplots(4, 8, figsize=(DOUBLE_COL, DOUBLE_COL * 0.6),
                                 gridspec_kw=dict(wspace=0.06, hspace=0.42))
        for i, (D, P, lab) in enumerate(SNAP_CONDS):
            for j, s in enumerate(SNAP_SEEDS):
                ax = axes[i, j]
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.4); sp.set_color("#999999")
                t = tt[(D, P)]
                f = SNAP_ROOT / "intended" / f"D{D}_P{P:.1f}" / f"seed_{s}_t{t}.npy"
                if t is not None and f.exists():
                    ax.imshow(render_hue(np.load(f)), interpolation="nearest", origin="upper")
                    r = st[(st.D == D) & (st.P == P) & (st.seed == s) & (st.t == t)].iloc[0]
                    why = r.excluded_because
                    ax.set_title(f"seed {s}, C = {fmt_C(r.C_run)}\n" + (f"excl.: {why}" if why else "candidate"),
                                 fontsize=4.3, pad=1.5, color="#8a8a8a" if why else "#1a1a1a", linespacing=1.15)
                    if (D, P) in chosen and int(chosen[(D, P)].seed) == s and int(chosen[(D, P)].t) == t:
                        for sp in ax.spines.values():
                            sp.set_linewidth(1.6); sp.set_color("black")
                if j == 0:
                    ax.set_ylabel(f"{lab}\nt = {t}", fontsize=5.5)
        fig.text(0.5, 0.06, f"Candidates: successful runs with >= {SNAP_MIN_ACTIVE} active cells at the snapshot "
                 "time; framed: the candidate whose C is closest to the condition median C (shown in Fig. 7).",
                 ha="center", fontsize=5.5, color="#444444")
        savefig(fig, out, "figS_snapshots_all_seeds", png_dpi=200)
    return sel, st, dom


# =============================== report ===============================
def tex_lines():
    return TEX.read_text().splitlines() if TEX.exists() else []


def locate(lines, line_no, snippet):
    """Return the 1-based line holding snippet (the expected one if it still matches)."""
    if 0 < line_no <= len(lines) and snippet in lines[line_no - 1]:
        return line_no, ""
    for i, l in enumerate(lines, 1):
        if snippet in l:
            return i, f" (moved from l.{line_no})"
    return line_no, " (snippet NOT found in the tex; check)"


def ck(k):
    """(D, P) key (tuple or Series) as plain text."""
    k = tuple(k.to_numpy()) if isinstance(k, pd.Series) else tuple(k)
    return f"({int(k[0])}, {float(k[1]):.1f})"


def pf(p):
    if not np.isfinite(p):
        return "n/a"
    if p == 0:
        return "< 1e-300"  # below double precision
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def main():
    global SD_ROOT, SNAP_ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results" / "npj_rev" / "ext1"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--sweep-root", default=str(ROOT / "results" / "spatial_dyadic"),
                    help="spatial_dyadic sweep (intended/, as_coded/, sayama_lba_fig1.csv)")
    ap.add_argument("--snap-root", default=str(ROOT / "results" / "spatial_dyadic_snapshots"),
                    help="snapshot configs (intended/D*_P*/seed_*_t*.npy)")
    args = ap.parse_args()
    SD_ROOT, SNAP_ROOT = Path(args.sweep_root), Path(args.snap_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    parts, curves = [], {}
    for rule in ("intended", "as_coded"):
        df, cv = load_rule(rule)
        parts.append(df)
        curves.update(cv)
    runs = pd.concat(parts, ignore_index=True)
    runs.to_csv(out / "per_run.csv", index=False)
    n_cond = runs.groupby("rule").apply(lambda g: g.groupby(["D", "P"]).ngroups, include_groups=False)
    print("conditions loaded:", n_cond.to_dict(), "| runs:", runs.groupby("rule").size().to_dict())

    tab, boots = condition_table(runs, curves, args.n_boot, rng)
    tab.to_csv(out / "conditions.csv", index=False)
    ti = tab[tab.rule == "intended"]
    best_row = ti.sort_values("ABC", ascending=False).iloc[0]
    best = (int(best_row.D), float(best_row.P))

    # ---- (d) statistics
    datasets = {"intended": regression_data(runs, "intended"),
                "as_coded": regression_data(runs, "as_coded")}
    if rsd.SAYAMA_CSV.exists():
        datasets["mathematica"] = mathematica_data()
    else:
        print("Private Mathematica comparison CSV absent: generating all main figures and JAX statistics; "
              "the archived reference-comparison statistics are supplied separately.", flush=True)
    reg, joint = run_regressions(datasets)
    reg.to_csv(out / "regression.csv", index=False)
    joint.to_csv(out / "joint_tests.csv", index=False)
    tr = [trend_tests(runs, r) for r in ("intended", "as_coded")]
    trends = pd.concat([x[0] for x in tr], ignore_index=True)
    strata = pd.concat([x[1] for x in tr], ignore_index=True)
    trends.to_csv(out / "trends.csv", index=False)
    strata.to_csv(out / "trends_strata.csv", index=False)
    c_best, c_all, cP, cD, _ = contrasts(runs, tab, "intended")
    c_best.to_csv(out / "contrasts_original_vs_best.csv", index=False)
    c_all.to_csv(out / "contrasts_original_vs_all.csv", index=False)
    cP.to_csv(out / "contrasts_P0_vs_P.csv", index=False)
    cD.to_csv(out / "contrasts_D100_vs_D.csv", index=False)

    # ---- (e) failures, sensitivity, bridge
    fail = tab[["rule", "D", "P", "n_runs"] + [f"n_{c}" for c in CATS]]
    fail.to_csv(out / "failures.csv", index=False)
    fail.groupby(["rule", "P"])[[f"n_{c}" for c in CATS] + ["n_runs"]].sum().to_csv(out / "failures_by_P.csv")
    fail.groupby(["rule", "D"])[[f"n_{c}" for c in CATS] + ["n_runs"]].sum().to_csv(out / "failures_by_D.csv")
    sens_rows = []
    for (rule, D, P), g in runs.groupby(["rule", "D", "P"]):
        for variant, mask in (("primary (success, runaway excluded)", g.used),
                              ("runaway runs included", g.sayama_ok),
                              ("all non-extinct runs (no log filter)", g.n_active_final > 0)):
            gg = g[mask]
            sens_rows.append(dict(rule=rule, D=D, P=P, variant=variant, n=len(gg),
                                  B=float(np.nanmean(gg.B_run)) if len(gg) else np.nan,
                                  C=float(gg.C_run.mean()) if len(gg) else np.nan))
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(out / "sensitivity.csv", index=False)
    gi = runs[runs.rule == "intended"]
    r10, n10 = int(gi[gi.D == 10].runaway.sum()), int((gi.D == 10).sum())
    r20, n20 = int(gi[gi.D >= 20].runaway.sum()), int((gi.D >= 20).sum())
    bridge_p = stats.fisher_exact([[r10, n10 - r10], [r20, n20 - r20]])[1]
    ga = runs[(runs.rule == "as_coded") & (runs.P > 0)]
    a10, an10 = int(ga[ga.D == 10].runaway.sum()), int((ga.D == 10).sum())
    a20, an20 = int(ga[ga.D >= 20].runaway.sum()), int((ga.D >= 20).sum())

    # ---- figures
    fig5(tab, out, best)
    fig6(tab, out, best)
    fig_trends(tab, boots, out)
    fig_failures(runs, out)
    sel, snapst, snapdom = snapshots(runs, out)

    # ---- (f) summary
    write_summary(out, runs, tab, boots, reg, joint, trends, strata, c_best, c_all, cP, cD, best, sens, sel,
                  snapst, snapdom, (r10, n10, r20, n20, bridge_p, a10, an10, a20, an20), args)
    print(f"wrote {out}")


def strata_desc(sd, rule, var, measure):
    """Per-stratum Spearman of measure vs var: (text, positive levels, negative levels, n.s. levels)."""
    s = sd[(sd.rule == rule) & (sd.trend_in == var) & (sd.measure == measure)]
    strat = "D" if var == "P" else "P"
    f = (lambda v: f"{v:.1f}") if strat == "P" else (lambda v: f"{int(v)}")
    sig = s.varies & (s.p < 0.05)
    pos = [f(v) for v in s[sig & (s.rho > 0)].level]
    neg = [f(v) for v in s[sig & (s.rho < 0)].level]
    ns = [f(v) for v in s[s.varies & ~(s.p < 0.05)].level]
    const = [f(v) for v in s[~s.varies].level]
    parts = [f"significantly positive at {strat} = {', '.join(pos)}" if pos else "significantly positive at no "
             + strat, f"negative at {strat} = {', '.join(neg)}" if neg else "negative at none"]
    if ns:
        parts.append(f"n.s. at {strat} = {', '.join(ns)}")
    if const:
        parts.append(f"no variation at {strat} = {', '.join(const)}")
    return "; ".join(parts), pos, neg, ns


def effect(r, invert=False):
    """'raises' / 'lowers' / 'does not significantly change' from a pooled Spearman row of trends.csv;
    invert=True for locality (= smaller D)."""
    if not (r.spearman_p < 0.05):
        return "does not significantly change"
    return "raises" if (r.spearman_rho > 0) != invert else "lowers"


def locality_effect(pos, neg, ns, name):
    """Effect of locality (smaller D) on measure `name` from its per-P strata vs D (pos: rises with D)."""
    if pos and not neg:
        return f"lowers {name} at P = {', '.join(pos)}" + (f" (n.s. at P = {', '.join(ns)})" if ns else "")
    if neg and not pos:
        return f"raises {name} at P = {', '.join(neg)}" + (f" (n.s. at P = {', '.join(ns)})" if ns else "")
    if pos and neg:
        return f"raises {name} at P = {', '.join(neg)} and lowers it at P = {', '.join(pos)}"
    return f"does not significantly change {name} at any P"


def write_summary(out, runs, tab, boots, reg, joint, trends, strata, c_best, c_all, cP, cD, best, sens, sel,
                  snapst, snapdom, bridge, args):
    ti = tab[tab.rule == "intended"].set_index(["D", "P"])
    gi = runs[runs.rule == "intended"]
    o = ti.loc[ORIGINAL]
    b = ti.loc[best]
    lines = tex_lines()
    md = []
    w = md.append
    w("# Extension I (Sec. 3) with the intended dyadic rule: numbers for the revision\n")
    w(f"Generated by `scripts/npj_rev_ext1.py` (bootstrap: {args.n_boot} resamples, seed {args.seed}). "
      f"Data: `results/spatial_dyadic/intended` (JAX, winner_rule='original', score='mix', L=100, 2000 steps), "
      f"{len(gi)} runs in {gi.groupby(['D', 'P']).ngroups} conditions "
      f"(runs per condition: {gi.groupby(['D', 'P']).size().min()}-{gi.groupby(['D', 'P']).size().max()}). "
      "P = 0 runs are shared by both dyadic rules. Every finding below is computed from the data at run time.\n")
    w("## Definitions used\n")
    w("- **Time convention** (`compare_original_schc.run_batch`): CSV row t holds `n_active` after t steps but the "
      "events (replications, patterns, log bytes) of the (t+1)-th step. Row 1999 therefore gives C and the log size "
      "after all 2000 steps (t = 2000, end of run), and rows 1500-1999 cover steps 1501-2000 (t = 1500-2000, as in "
      "the Fig. 5 caption).")
    w("- **Success** (Sayama's criterion): 1 MB < cumulative log size at the end of the run < 300 MB, at least one "
      "active cell in the last row, and a complete 2000-step record.")
    nonrun, isrun = runs[~runs.runaway].max_rep_size_max, runs[runs.runaway].max_rep_size_max
    if len(isrun):
        w(f"- **Runaway**: some replication event copies >= {RUNAWAY_SIZE} cells (10% of the grid). Runaway runs count "
          f"as failures. Among non-runaway runs the largest event is {int(nonrun.max())} cells and runaway runs reach "
          f"{int(isrun.min())}-{int(isrun.max())}, so any threshold in ({int(nonrun.max())}, {int(isrun.min())}] "
          "gives the same classification.")
    else:
        w(f"- **Runaway**: some replication event copies >= {RUNAWAY_SIZE} cells; no run reaches it (largest event "
          f"{int(nonrun.max())} cells).")
    n300 = int((runs.log_MB * 1e6 >= LOG_HI).sum())
    rl = runs[runs.runaway].log_MB
    if n300 == 0:
        w(f"- In the JAX runs no log reaches 300 MB (maximum {runs.log_MB.max():.0f} MB"
          + (f"; runaway runs {rl.min():.0f}-{rl.max():.0f} MB). The runaway runs would therefore pass the log filter, "
             "and the size criterion is what excludes them (the Mathematica explosive run was excluded by its 380 MB "
             "log)." if len(rl) else ")."))
    else:
        w(f"- {n300} JAX runs reach the 300 MB log threshold (maximum {runs.log_MB.max():.0f} MB).")
    w("- **A** = successful runs of 30 (Wilson 95% interval); **B** = per-step ensemble mean of `mean_rep_size` over "
      "successful runs, averaged over the final 500 steps (t = 1500-2000); **C** = mean over successful runs of the "
      "cumulative pattern types at the end of the run (t = 2000); 95% intervals for B and C: percentile bootstrap "
      "over the successful runs of the condition; for A x B x C: bootstrap over all 30 runs.")
    w("- **For the Methods** (state this in the manuscript): the B and C intervals are *conditional on success* "
      "(resampling only the successful runs of a condition), whereas `analyze_spatial_dyadic.bootstrap` resamples all "
      "runs; A x B x C resamples all runs. Means over D or P: A with a Wilson interval on the pooled counts (scaled "
      "to runs per condition; a bootstrap would give the zero-width interval [30, 30] where every run succeeds), "
      "B, C and A x B x C with the mean of the per-condition bootstrap replicates.")
    nan_gap = (tab.B - tab.B_mean_of_runs).abs().max()
    nan_runs = runs[runs.n_nan_final500 > 0]
    w(f"- B as the ensemble curve average equals the mean of per-run B (max difference {nan_gap:.1e}); the per-run "
      f"values are what the tests and regressions use. Runs with NaN `mean_rep_size` in the final 500 rows: "
      f"{len(nan_runs)} (categories: {', '.join(sorted(nan_runs.category.unique())) or 'none'}; used: "
      f"{int(nan_runs.used.sum())}).\n")

    # ---------------- Fig 5
    w("## (a) Fig. 5 replacement\n")
    w("Files: `fig5_heatmaps.pdf/png` (2x2 with colourbars) and the single panels `P-D-final-num-runs`, "
      "`P-D-final-size`, `P-D-final-nctypes`, `P-D-overall-performance` (.pdf/.png; same base names as the "
      "manuscript's files; about 92 mm wide, so include them with `width=0.49\\textwidth` rather than "
      "`height=1.4in`). Panel titles: B over t = 1500-2000, C at t = 2000 (end of run), as in the caption.\n")
    w("**Orientation**: columns D = 10 ... 100 from left to right; rows P = 0 (top) ... 1 (bottom). "
      "The original setting (D, P) = (L, 0) is the **top-right** cell in every panel (solid box), so the "
      "manuscript's 'top right corner' wording stays correct. Panel (d) also marks the best condition "
      "(dashed box). Both boxes are black lines on a white halo, visible on the darkest cells. Annotations: A as "
      "counts, B with two decimals, C in thousands, A x B x C in millions. "
      f"Colour map: seaborn '{HEAT_CMAP}' (sequential, colour-blind safe), one colourbar per panel.\n")
    w("| measure | min (condition) | max (condition) | original (L,0) [95% CI] |")
    w("|---|---|---|---|")
    for key, nm in (("A", "A"), ("B", "B"), ("C", "C"), ("ABC", "A x B x C")):
        mn, mx = ti[key].idxmin(), ti[key].idxmax()
        lo, hi = {"A": ("A_wilson_lo", "A_wilson_hi"), "B": ("B_lo", "B_hi"), "C": ("C_lo", "C_hi"),
                  "ABC": ("ABC_lo", "ABC_hi")}[key]
        f = (lambda v: f"{v:.3g}") if key != "A" else (lambda v: f"{v:.0f}")
        w(f"| {nm} | {f(ti[key].min())} {ck(mn)} | {f(ti[key].max())} {ck(mx)} | {f(o[key])} [{o[lo]:.3g}, {o[hi]:.3g}] |")
    w("")
    w(f"Best condition by A x B x C: (D, P) = {ck(best)}: A = {b.A:.0f}, B = {b.B:.2f}, C = {b.C:.0f}, A x B x C = "
      f"{b.ABC / 1e6:.2f}e6 [{b.ABC_lo / 1e6:.2f}, {b.ABC_hi / 1e6:.2f}]e6, i.e. {b.ABC / o.ABC:.1f} times the "
      f"original ({o.ABC / 1e6:.2f}e6 [{o.ABC_lo / 1e6:.2f}, {o.ABC_hi / 1e6:.2f}]e6). "
      f"The original ranks {int(o.rank_ABC)} of {len(ti)} by A x B x C.")
    top = ti.sort_values("ABC", ascending=False).head(10)
    w("\nTop 10 by A x B x C (all at P >= " + f"{top.index.get_level_values('P').min():.1f}):\n")
    w("| rank | (D, P) | A [Wilson] | B [CI] | C [CI] | A x B x C (x10^6) [CI] |")
    w("|---|---|---|---|---|---|")
    for k, r in top.iterrows():
        w(f"| {int(r.rank_ABC)} | {ck(k)} | {r.A:.0f} [{r.A_wilson_lo:.1f}, {r.A_wilson_hi:.1f}] | {r.B:.2f} "
          f"[{r.B_lo:.2f}, {r.B_hi:.2f}] | {r.C:.0f} [{r.C_lo:.0f}, {r.C_hi:.0f}] | {r.ABC / 1e6:.2f} "
          f"[{r.ABC_lo / 1e6:.2f}, {r.ABC_hi / 1e6:.2f}] |")
    rk = stats.spearmanr(ti.ABC, ti.composite_rank)[0]
    rk2 = stats.spearmanr(ti.ABC, ti.ABgmC)[0]
    best_gm = ti.ABgmC.idxmax()
    w(f"\nRobustness of the ranking: Spearman rho between A x B x C and the equal-weight mean percentile rank of "
      f"(A, B, C) = {rk:.2f}; with C replaced by its geometric mean over runs (log-C scale) rho = {rk2:.2f}, best "
      f"condition {ck(best_gm)}, original ranked {int(ti.loc[ORIGINAL].rank_ABgmC)}.\n")

    # per-P pooled table
    w("Mean over D (95% interval: A Wilson on pooled counts; B, C, A x B x C bootstrap), intended rule:\n")
    w("| P | A | B | C | A x B x C (x10^6) |")
    w("|---|---|---|---|---|")
    for P in P_VALUES:
        cells = []
        for m in ("A", "B", "C", "ABC"):
            v = pooled(tab, boots, "intended", "P", P, m)
            sc = {"C": 1, "ABC": 1e6}.get(m, 1)
            fm = "{:.0f}" if m == "C" else "{:.2f}"
            cells.append(f"{fm.format(v[0] / sc)} [{fm.format(v[1] / sc)}, {fm.format(v[2] / sc)}]")
        w(f"| {P:.1f} | " + " | ".join(cells) + " |")
    ref = pd.read_csv(SD_ROOT / "sayama_lba_fig1.csv")
    ref["P"] = ref.P.round(1)
    w("\nPublished values (Mathematica, swap typo; `results/spatial_dyadic/sayama_lba_fig1.csv`) vs intended rule, "
      "mean over D:\n")
    w("| P | A Mathematica | A intended | B Mathematica | B intended | C Mathematica | C intended |")
    w("|---|---|---|---|---|---|---|")
    for P in P_VALUES:
        m = ref[ref.P == P]
        w(f"| {P:.1f} | {m.successful_runs.mean():.1f} | {pooled(tab, boots, 'intended', 'P', P, 'A')[0]:.1f} | "
          f"{m.mean_rep_size_final500.mean():.2f} | {pooled(tab, boots, 'intended', 'P', P, 'B')[0]:.2f} | "
          f"{m.cum_pattern_types_final.mean():.0f} | {pooled(tab, boots, 'intended', 'P', P, 'C')[0]:.0f} |")
    mo = ref[(ref.D == 100) & (ref.P == 0.0)].iloc[0]
    w(f"\nOriginal (100, 0): Mathematica A = {mo.successful_runs:.0f}, B = {mo.mean_rep_size_final500:.2f}, "
      f"C = {mo.cum_pattern_types_final:.0f}; intended (JAX) A = {o.A:.0f}, B = {o.B:.2f}, C = {o.C:.0f}. "
      f"Mathematica maxima: B at {ck(ref.loc[ref.mean_rep_size_final500.idxmax(), ['D', 'P']])}, C at "
      f"{ck(ref.loc[ref.cum_pattern_types_final.idxmax(), ['D', 'P']])}; intended: B at {ck(ti.B.idxmax())}, "
      f"C at {ck(ti.C.idxmax())}.")
    w("\nMean over P (95% interval as above), intended rule:\n")
    w("| D | A | B | C |")
    w("|---|---|---|---|")
    for D in D_VALUES:
        cells = []
        for m in ("A", "B", "C"):
            v = pooled(tab, boots, "intended", "D", D, m)
            fm = "{:.0f}" if m == "C" else "{:.2f}"
            cells.append(f"{fm.format(v[0])} [{fm.format(v[1])}, {fm.format(v[2])}]")
        w(f"| {D} | " + " | ".join(cells) + " |")
    w("\nFull per-condition table with intervals: `conditions.csv` (rule, D, P, n_runs, A, A_wilson_lo/hi (counts), "
      "B, B_lo/hi, C, C_lo/hi, ABC, ABC_lo/hi, ABC_rel, ranks, failure counts). Trends with per-D/per-P lines "
      "and pooled intervals: `figS_trends.pdf`.\n")

    # ---------------- Fig 6
    labs = label_set(best)
    w("## (b) Fig. 6 replacement\n")
    w("Files: `fig6_3d.pdf/png` (3-D scatter, one point per (D, P)) and `fig6_pairs.pdf/png` (the same data as "
      "three 2-D projections; recommended, because no point is hidden and labels do not overlap). "
      "**Colour metric**: relative overall performance A x B x C / max over the 60 conditions (the quantity of "
      f"Fig. 5d, rescaled to [0, 1]); colour map '{HEAT_CMAP}' with a colourbar. The old ideal point (30, 6, 11000) "
      f"is no longer meaningful (C now reaches {ti.C.max():.0f}). Labelled points (black outline): "
      f"{'; '.join(labs.values())}, i.e. the Fig. 7 conditions and the best condition by A x B x C. "
      f"**In `fig6_pairs`, A (an integer count) is jittered horizontally by up to +/-{A_JITTER}** so that "
      "conditions with equal A do not overlap; the figure states this, and the caption should too. "
      f"The other top conditions by A x B x C ({', '.join(ck(k) for k in top.index[1:4])}) lie next to the best "
      "one and are not labelled separately.\n")

    # ---------------- statistics
    w("## (d) R1.1 statistics\n")
    w("### Sayama's quadratic model y = a + bD + cD^2 + dP + eP^2 + fDP\n")
    w("A: OLS on the 60 per-condition counts (as in Sayama's analysis) and logistic regression on per-run success; "
      "B, C, log C: OLS on per-run values of successful runs. p: classical; p_HC3: heteroskedasticity-robust. "
      "Full table (also as_coded and Mathematica): `regression.csv`; block tests: `joint_tests.csv`.\n")
    ri = reg[reg.data == "intended"]
    for (resp, model), g in ri.groupby(["response", "model"], sort=False):
        w(f"**{resp} ({model}, n = {g.n.iloc[0]}, {'McFadden R^2' if model == 'logit' else 'R^2'} = {g.r2.iloc[0]:.3f})**: "
          + "; ".join(f"{r.term}: {r.coef:.4g} (p {pf(r.p)}" + (f", HC3 {pf(r.p_hc3)})" if np.isfinite(r.p_hc3) else ")")
                      for r in g.itertuples()))
        w("")
    w("Block tests (intended): \n")
    w("| response | model | D terms (D, D^2, DP) p [HC3] | P terms (P, P^2, DP) p [HC3] | DP p [HC3] |")
    w("|---|---|---|---|---|")
    ji = joint[joint.data == "intended"]
    for (resp, model), g in ji.groupby(["response", "model"], sort=False):
        g = g.set_index("block")
        cell = lambda k: f"{pf(g.loc[k, 'p'])}" + (f" [{pf(g.loc[k, 'p_hc3'])}]" if np.isfinite(g.loc[k, 'p_hc3']) else "")
        w(f"| {resp} | {model} | {cell('D terms (D, D^2, DP)')} | {cell('P terms (P, P^2, DP)')} | {cell('DP')} |")
    okB = gi[gi.used].groupby("P").B_run.var()
    w(f"\nThe classical and HC3 p of the B blocks differ because the variance of per-run B falls with P "
      f"(from {okB.max():.3g} to {okB.min():.3g}, a {okB.max() / okB.min():.0f}-fold range); HC3 is the one to quote.")
    w("")
    w("Stationary points of the fitted surfaces (outside the sampled range = monotone over it):\n")
    for resp in ("B", "C", "logC"):
        c = ri[(ri.response == resp) & (ri.model == "OLS")].set_index("term").coef
        vP = -(c["P"] + c["DP"] * 55.0) / (2 * c["P^2"])
        vD = -(c["D"] + c["DP"] * 0.5) / (2 * c["D^2"])
        kP = "maximum" if c["P^2"] < 0 else "minimum"
        kD = "maximum" if c["D^2"] < 0 else "minimum"
        inP = "inside [0, 1]" if 0 <= vP <= 1 else "outside [0, 1]"
        inD = "inside [10, 100]" if 10 <= vD <= 100 else "outside [10, 100]"
        w(f"- {resp}: in P (at D = 55) a {kP} at P = {vP:.2f} ({inP}); in D (at P = 0.5) a {kD} at D = {vD:.0f} ({inD}).")
    w("")

    w("### Monotone trends (`trends.csv`, per stratum: `trends_strata.csv`)\n")
    w("P-trend pooled over D (and D-trend pooled over P). Spearman rho on per-run values (A as 0/1 per run); "
      "slope from OLS with the other parameter as a factor (HC3 p); per-stratum Spearman: number of strata with "
      "a significant positive / negative rho (p < 0.05), and Stouffer's combined p. A also: Cochran-Armitage on "
      "pooled counts.\n")
    w("| trend in | measure | n | rho (p) | adj. slope (HC3 p) | strata +/- sig. | Stouffer p | C-A p |")
    w("|---|---|---|---|---|---|---|---|")
    for r in trends[trends.rule == "intended"].itertuples():
        ca = pf(r.cochran_armitage_p) if hasattr(r, "cochran_armitage_p") and np.isfinite(r.cochran_armitage_p) else ""
        w(f"| {r.trend_in} | {r.measure} | {r.n} | {r.spearman_rho:.2f} ({pf(r.spearman_p)}) | {r.slope_adj:.4g} "
          f"({pf(r.slope_p_hc3)}) | {r.strata_pos_sig}/{r.strata_neg_sig} of {r.n_strata} | {pf(r.stouffer_p)} | {ca} |")
    w("\nPer-stratum Spearman rho (* p < 0.05; strata with no variation omitted):\n")
    for r in trends[trends.rule == "intended"].itertuples():
        w(f"- {r.measure} vs {r.trend_in}: {r.strata_rho}")
    w("")

    w("### Contrasts\n")
    w(f"**Original (100, 0) vs best by A x B x C {ck(best)}** (`contrasts_original_vs_best.csv`; Holm over the 3 tests; "
      "Cliff's delta = P(best > original) - P(best < original); for A it equals the difference in success fraction):\n")
    w("| measure | best | original | test | p | p Holm | Cliff's delta |")
    w("|---|---|---|---|---|---|---|")
    for r in c_best.itertuples():
        w(f"| {r.measure} | {r.stat_x:.3g} (n={r.n_x}) | {r.stat_y:.3g} (n={r.n_y}) | {r.test} | {pf(r.p)} | "
          f"{pf(r.p_holm)} | {r.cliffs_delta:+.2f} |")
    w("\n**Each of the 59 other conditions vs the original** (`contrasts_original_vs_all.csv`, Holm within measure):\n")
    for m in ("A", "B", "C"):
        g = c_all[c_all.measure == m]
        hi_ = g[(g.p_holm < 0.05) & (g.cliffs_delta > 0)]
        lo_ = g[(g.p_holm < 0.05) & (g.cliffs_delta < 0)]
        raw_hi = g[(g.p < 0.05) & (g.cliffs_delta > 0)]
        w(f"- {m}: higher than the original in {len(hi_)}/{len(g)} conditions after Holm (unadjusted p<0.05: "
          f"{len(raw_hi)}), lower in {len(lo_)}; point estimate higher in {(g.cliffs_delta > 0).sum()}/{len(g)}."
          + (f" Significantly higher after Holm: {', '.join(f'({int(r.D)},{r.P:.1f})' for r in hi_.itertuples())}."
             if 0 < len(hi_) <= 30 else ""))
    w(f"\n**P = 0 vs each P > 0, pooled over D** (`contrasts_P0_vs_P.csv`; Holm over all {len(cP)} tests):\n")
    w("| P | measure | P value | P=0 value | p | p Holm | Cliff's delta |")
    w("|---|---|---|---|---|---|---|")
    for r in cP.itertuples():
        w(f"| {r.x} | {r.measure} | {r.stat_x:.3g} | {r.stat_y:.3g} | {pf(r.p)} | {pf(r.p_holm)} | {r.cliffs_delta:+.2f} |")
    w(f"\n**D = 100 vs each D < 100, pooled over P** (`contrasts_D100_vs_D.csv`; Holm over all {len(cD)} tests). "
      f"Caveat for B: pooling over P mixes strata whose B variance differs {okB.max() / okB.min():.0f}-fold and whose "
      "D effect differs in sign; the per-P Spearman results above are the better evidence for a D effect on B.\n")
    w("| D | measure | D value | D=100 value | p | p Holm | Cliff's delta |")
    w("|---|---|---|---|---|---|---|")
    for r in cD.itertuples():
        w(f"| {r.x} | {r.measure} | {r.stat_x:.3g} | {r.stat_y:.3g} | {pf(r.p)} | {pf(r.p_holm)} | {r.cliffs_delta:+.2f} |")
    w("")
    w("### log(C) robustness\n")
    cC = ri[(ri.response == "C") & (ri.model == "OLS")].set_index("term")
    cL = ri[(ri.response == "logC") & (ri.model == "OLS")].set_index("term")
    DD, PP = np.meshgrid(D_VALUES, P_VALUES)
    dP = lambda c: c["P"] + 2 * c["P^2"] * PP + c["DP"] * DD
    dD = lambda c: c["D"] + 2 * c["D^2"] * DD + c["DP"] * PP
    agP = np.mean(np.sign(dP(cC.coef)) == np.sign(dP(cL.coef)))
    agD = np.mean(np.sign(dD(cC.coef)) == np.sign(dD(cL.coef)))
    fitC = (rsd.design(DD.ravel(), PP.ravel()) @ cC.coef[TERMS].to_numpy())
    fitL = (rsd.design(DD.ravel(), PP.ravel()) @ cL.coef[TERMS].to_numpy())
    jC = joint[(joint.data == "intended") & (joint.response == "C")].set_index("block")
    jL = joint[(joint.data == "intended") & (joint.response == "logC")].set_index("block")
    w(f"- Quadratic model on log C (R^2 = {cL.r2.iloc[0]:.3f}; C: {cC.r2.iloc[0]:.3f}). Individual polynomial terms are "
      "collinear and not comparable between scales, so the check is on the fitted surface: the sign of the fitted "
      f"dy/dP agrees between C and log C at {100 * agP:.0f}% of the 60 grid points and the sign of dy/dD at "
      f"{100 * agD:.0f}%; Spearman rho between the fitted condition values {stats.spearmanr(fitC, fitL)[0]:.2f}. "
      f"Block tests (HC3): D terms p = {pf(jC.loc['D terms (D, D^2, DP)', 'p_hc3'])} (C) vs "
      f"{pf(jL.loc['D terms (D, D^2, DP)', 'p_hc3'])} (log C); P terms p = {pf(jC.loc['P terms (P, P^2, DP)', 'p_hc3'])} "
      f"vs {pf(jL.loc['P terms (P, P^2, DP)', 'p_hc3'])}.")
    tC = trends[(trends.rule == "intended") & (trends.measure == "logC")]
    for r in tC.itertuples():
        w(f"- log C trend in {r.trend_in}: adjusted slope {r.slope_adj:.3g} (HC3 p {pf(r.slope_p_hc3)}), "
          f"{r.strata_pos_sig}/{r.strata_neg_sig} strata significantly +/-.")
    w(f"- Ranking with geometric-mean C: best {ck(best_gm)}, Spearman rho with the A x B x C ranking {rk2:.2f} (above). "
      "Rank-based tests (Mann-Whitney, Spearman) are unchanged by the log transform.\n")

    # ---------------- R1.2
    w("## (e) R1.2: failure breakdown\n")
    w("Mutually exclusive categories, assigned in the order incomplete > extinct > runaway > log >= 300 MB > "
      "no growth (log <= 1 MB) > success. Per condition: `failures.csv`; pooled: `failures_by_P.csv`, "
      "`failures_by_D.csv`; figure `figS_failures.pdf` (each column shares its y-axis between the two rules).\n")
    w("| rule | runs | success | extinct | no growth | runaway | log >= 300 MB | incomplete |")
    w("|---|---|---|---|---|---|---|---|")
    for rule, g in runs.groupby("rule", sort=False):
        vc = g.category.value_counts()
        w(f"| {rule} | {len(g)} | " + " | ".join(str(int(vc.get(c, 0))) for c in CATS) + " |")
    w("\n(as_coded includes the shared P = 0 runs.)\n")
    for rule in ("intended", "as_coded"):
        g = runs[(runs.rule == rule) & (runs.category != "success")]
        for c in ("extinct", "runaway"):
            gg = g[g.category == c]
            w(f"- {rule}, {c}: " + (", ".join(f"(D={r.D}, P={r.P:.1f}, seed {r.seed})" for r in gg.itertuples())
                                     or "none"))
        ng = g[g.category == "no_growth"]
        w(f"- {rule}, no growth: {len(ng)} runs; by P (pooled over D): "
          + ", ".join(f"P={P:.1f}: {int((ng.P == P).sum())}" for P in P_VALUES)
          + (f"; final active cells median {ng.n_active_final.median():.0f} (max {ng.n_active_final.max()})."
             if len(ng) else "."))
    w("")
    w("**Sensitivity to the exclusions** (`sensitivity.csv`): B and C per condition recomputed (i) with the runaway "
      "runs included and (ii) over all non-extinct runs (no log filter).\n")
    si = sens[sens.rule == "intended"].pivot_table(index=["D", "P"], columns="variant", values=["B", "C"])
    prim = "primary (success, runaway excluded)"
    for v in ("runaway runs included", "all non-extinct runs (no log filter)"):
        dB = (si["B"][v] - si["B"][prim]).abs()
        dC = (si["C"][v] - si["C"][prim]).abs() / si["C"][prim]
        changed = dB[dB > 1e-12].index.tolist()
        w(f"- {v}: B changes in {len(changed)} conditions (max |dB| = {dB.max():.3g} at {ck(dB.idxmax())}); "
          f"C changes by at most {100 * dC.max():.1f}% (at {ck(dC.idxmax())}).")
    w("\nMean over D by P under each variant (intended):\n")
    w("| P | B primary | B incl. runaway | B all non-extinct | C primary | C incl. runaway | C all non-extinct |")
    w("|---|---|---|---|---|---|---|")
    sp = sens[sens.rule == "intended"].groupby(["P", "variant"])[["B", "C"]].mean()
    vs = [prim, "runaway runs included", "all non-extinct runs (no log filter)"]
    for P in P_VALUES:
        w(f"| {P:.1f} | " + " | ".join(f"{sp.loc[(P, v), 'B']:.2f}" for v in vs) + " | "
          + " | ".join(f"{sp.loc[(P, v), 'C']:.0f}" for v in vs) + " |")
    r10, n10, r20, n20, bp, a10, an10, a20, an20 = bridge
    w("\n**Sec. 4 bridge statistic (l.300), intended rule only**: runaway runs at D = 10: "
      f"{r10}/{n10} ({', '.join(f'P={r.P:.1f}' for r in gi[(gi.D == 10) & gi.runaway].itertuples()) or 'none'}); at "
      f"D >= 20: {r20}/{n20}; Fisher exact two-sided p = {pf(bp)}. (For comparison, as_coded P > 0 only: {a10}/{an10} "
      f"vs {a20}/{an20}; not to be pooled with the intended runs.)\n")

    # ---------------- snapshots
    dom = snapdom if len(snapdom) else pd.DataFrame(columns=["D", "P", "t"])
    w("## (c) Fig. 7 replacement\n")
    w("Rendering: L = 100 grid, cell type c coloured Hue[c/k] with k = 1000 (full saturation and brightness, as "
      "Sayama's MatrixPlot), empty cells white. Files: `fig7_snapshots.pdf/png`, single panels "
      "`snap_D*_P*_seed*_t*.png` (400x400 px), all seeds in `figS_snapshots_all_seeds.pdf` (chosen seed framed in "
      "black; non-candidates titled in grey with the reason), statistics in `snapshot_stats.csv` (all seeds and "
      "times) and `snapshot_dominance_by_time.csv` (candidates per condition and time).\n")
    w(f"**Choice rule**: candidates are seeds 0-7 that are successful (Sayama's criterion, not runaway) **and have >= "
      f"{SNAP_MIN_ACTIVE} active cells at the snapshot time** (a grown population; successful runs with only 26-126 "
      "cells exist and would not represent their condition). At the latest available time in (2000, 1750, 1500), "
      "take the candidate whose C is closest to the median C of all successful runs of the condition (ties: lower "
      "seed). The snapshot at t = 2000 is the state after 2000 steps (CSV row 1999 holds `n_active` after 1999 "
      "steps). The panel label gives the number of candidates and the number of *distinct patterns present in the "
      "snapshot* (not C).\n")
    w("Dominant-pattern share = fraction of active cells in components that are copies of the most abundant pattern "
      "(pattern = connected component, 8-neighbourhood, with its cell types, up to translation). Typical dominance is "
      "summarised over the **same candidates** (the population the panel is drawn from); a median over all seeds "
      "would include failed and near-empty grids, whose share is close to 1 almost by definition, and is shown "
      "only for reference.\n")
    w("| condition | seed | t | candidates (seeds) | C of run | condition median C | active cells | distinct patterns present | dominant-pattern share (rank among candidates) | candidates: median share [min, max]; cell-weighted |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in sel.itertuples():
        if r.n_candidates > 0:
            w(f"| {r.label} | {int(r.seed)} | {int(r.t)} | {r.n_candidates} ({r.candidate_seeds}) | {r.C_run:.0f} | "
              f"{r.cond_median_C:.0f} | {int(r.n_active)} | {int(r.n_patterns)} | {r.dominant_pattern_share:.2f} "
              f"({r.dom_rank_among_candidates} of {r.n_candidates}) | {r.cand_dom_median:.2f} [{r.cand_dom_min:.2f}, "
              f"{r.cand_dom_max:.2f}]; {r.cand_dom_cell_weighted:.2f} |")
        else:
            w(f"| {r.label} | pending | | 0 | | | | | | |")
    fb = sel[sel["candidate_rule"].astype(str).str.startswith("FALLBACK")] if "candidate_rule" in sel.columns \
        else sel.iloc[:0]
    if len(fb):
        w(f"\nFallback rule used (no successful seed with >= {SNAP_MIN_ACTIVE} cells) for: {', '.join(fb.label)}.")
    if len(dom) and "dom_median" in dom:
        w("\nDominance over time, candidates only (`snapshot_dominance_by_time.csv`):\n")
        w("| condition | t | candidates / seeds | dominant share: median [min, max] | cell-weighted | share > 0.5 | distinct patterns (median) | median over all non-empty seeds (not used) |")
        w("|---|---|---|---|---|---|---|---|")
        for r in dom.sort_values(["D", "P", "t"], ascending=[False, True, True]).itertuples():
            if r.n_candidates:
                w(f"| {r.label} | {r.t} | {r.n_candidates}/{r.n_seeds} | {r.dom_median:.2f} [{r.dom_min:.2f}, "
                  f"{r.dom_max:.2f}] | {r.dom_cell_weighted:.2f} | {r.n_dom_gt_half}/{r.n_candidates} | "
                  f"{r.patterns_median:.0f} | {r.dom_median_all_nonempty_seeds:.2f} |")
            else:
                w(f"| {r.label} | {r.t} | 0/{r.n_seeds} | | | | | {r.dom_median_all_nonempty_seeds:.2f} |")
    if len(snapst):
        w("\nSnapshot statistics over all available seeds (median [min, max]) at the latest available t:\n")
        w("| condition | t | n seeds | active cells | components | distinct patterns | dominant-pattern share | effective no. of cell types |")
        w("|---|---|---|---|---|---|---|---|")
        for D, P, lab in SNAP_CONDS:
            c = snapst[(snapst.D == D) & (snapst.P == P)]
            if not len(c):
                w(f"| {lab} | pending | 0 | | | | | |")
                continue
            c = c[c.t == c.t.max()]
            q = lambda s: f"{np.nanmedian(s):.3g} [{np.nanmin(s):.3g}, {np.nanmax(s):.3g}]"
            w(f"| {lab} | {int(c.t.iloc[0])} | {len(c)} | {q(c.n_active)} | {q(c.n_components)} | {q(c.n_patterns)} | "
              f"{q(c.dominant_pattern_share)} | {q(c.eff_n_types)} |")
        chk = snapst.dropna(subset=["csv_n_active_step_t"])
        w(f"\nConsistency with the sweep CSVs: snapshot active cells equal the CSV `n_active` at step t in "
          f"{int((chk.n_active == chk.csv_n_active_step_t).sum())}/{len(chk)} snapshots with a CSV row at t "
          f"(t = 2000 has no CSV row; step t-1 matches in "
          f"{int((snapst.n_active == snapst.csv_n_active_step_t_minus_1).sum())}/{len(snapst)}).\n")
        ok_sel = sel[sel.n_candidates > 0]
        if len(ok_sel):
            w("Suggested addition to the Fig. 7 caption: \"Each panel shows, at t = "
              + "/".join(sorted({str(int(v)) for v in ok_sel.t}))
              + f", the run (among the successful runs of seeds 0-7 with at least {SNAP_MIN_ACTIVE} active cells; "
              + ", ".join(f"{int(r.n_candidates)} in ({'abcd'[i]})" for i, r in enumerate(ok_sel.itertuples()))
              + ") whose cumulative number of pattern types is closest to the median of its condition. Numbers below "
              "the panels count the distinct patterns present in the snapshot.\"\n")
    else:
        w("\nNo snapshot configurations are available yet (results/spatial_dyadic_snapshots/intended).\n")

    # ---------------- text keyed to tex lines (every verdict below is derived from the data)
    w("## (f) Numbers for the Sec. 3 text, by sentence of sn-article-rev1.tex\n")
    tP = trends[(trends.rule == "intended")].set_index(["trend_in", "measure"])
    A_P = [pooled(tab, boots, "intended", "P", P, "A") for P in P_VALUES]
    B_P = [pooled(tab, boots, "intended", "P", P, "B") for P in P_VALUES]
    C_P = [pooled(tab, boots, "intended", "P", P, "C") for P in P_VALUES]
    B_D = [pooled(tab, boots, "intended", "D", D, "B") for D in D_VALUES]
    A_D = [pooled(tab, boots, "intended", "D", D, "A") for D in D_VALUES]
    C_D = [pooled(tab, boots, "intended", "D", D, "C") for D in D_VALUES]
    aA = c_all[c_all.measure == "A"]
    nA_hi = int(((aA.p_holm < 0.05) & (aA.cliffs_delta > 0)).sum())
    bB = c_all[c_all.measure == "B"]
    nB_hi = int(((bB.p_holm < 0.05) & (bB.cliffs_delta > 0)).sum())
    nB_lo = int(((bB.p_holm < 0.05) & (bB.cliffs_delta < 0)).sum())
    cC_ = c_all[c_all.measure == "C"]
    nC_hi = int(((cC_.p_holm < 0.05) & (cC_.cliffs_delta > 0)).sum())
    nABC_hi = int((ti.ABC_lo > o.ABC_hi).sum())
    jB = joint[(joint.data == "intended") & (joint.response == "B") & (joint.model == "OLS")].set_index("block")
    jA = joint[(joint.data == "intended") & (joint.response == "A") & (joint.model.str.startswith("logit"))].set_index("block")
    bmax = ti.B.idxmax()
    cmax = ti.C.idxmax()
    d10B = cD[(cD.x == f"D={D_VALUES[0]}") & (cD.measure == "B")]
    nDB = int(((cD.measure == "B") & (cD.x != f"D={D_VALUES[0]}") & (cD.p_holm < 0.05)).sum())
    nDB_all = int(((cD.measure == "B") & (cD.x != f"D={D_VALUES[0]}")).sum())
    # derived directions
    rA_P, rB_P, rC_P, rA_D = tP.loc[("P", "A")], tP.loc[("P", "B")], tP.loc[("P", "C")], tP.loc[("D", "A")]
    descB_D, posB_D, negB_D, nsB_D = strata_desc(strata, "intended", "D", "B")
    descC_D, posC_D, negC_D, nsC_D = strata_desc(strata, "intended", "D", "C")
    descA_D, posA_D, negA_D, nsA_D = strata_desc(strata, "intended", "D", "A")
    loc_B, loc_C = locality_effect(posB_D, negB_D, nsB_D, "B"), locality_effect(posC_D, negC_D, nsC_D, "C")
    loc_A = effect(rA_D, invert=True)
    dy_A, dy_B, dy_C = effect(rA_P), effect(rB_P), effect(rC_P)
    cP_means = np.array([s[0] for s in C_P])
    C_mono = bool(np.all(np.diff(cP_means) > 0))
    P_Cmax = P_VALUES[int(np.nanargmax(cP_means))]
    C_interior = P_VALUES[0] < P_Cmax < P_VALUES[-1]
    d_C10 = cP[(cP.x == f"P={P_VALUES[-1]}") & (cP.measure == "C")]
    cbm = c_best.set_index("measure")
    best_up = [m for m in cbm.index if cbm.loc[m, "p_holm"] < 0.05 and cbm.loc[m, "cliffs_delta"] > 0]
    best_dn = [m for m in cbm.index if cbm.loc[m, "p_holm"] < 0.05 and cbm.loc[m, "cliffs_delta"] < 0]
    run_D = sorted(gi[gi.runaway].D.unique().tolist())
    run_where = ("none" if not run_D else f"all at D = {run_D[0]}" if len(run_D) == 1
                 else f"at D = {', '.join(map(str, run_D))}")
    has_106 = bool(((gi.D == 10) & (gi.P == 0.6) & gi.runaway).any())
    pos = lambda s: f"{s[0]:.2f} [{s[1]:.2f}, {s[2]:.2f}]"
    lstr = lambda xs: ", ".join(xs)
    # snapshot-derived pieces
    def dom_row(D, P, t):
        r = dom[(dom.D == D) & (dom.P == P) & (dom.t == t)] if len(dom) else dom
        return r.iloc[0] if len(r) and r.iloc[0].n_candidates > 0 else None

    entries = [
        (107, "We show this extension substantially enhances",
         f"A x B x C of the best condition {ck(best)} is {b.ABC / o.ABC:.1f}x the original's; {nABC_hi}/59 conditions "
         f"have a bootstrap interval of A x B x C entirely above the original's. Best vs original (Holm p < 0.05): "
         f"higher {lstr(best_up) or 'none'}, lower {lstr(best_dn) or 'none'}. Across the grid dyadicity {dy_C} C and "
         f"{dy_B} B, so any enhancement is in success and exploration, not replicator size."),
        (137, "Locality promotes growth; dyadicity promotes diversity",
         f"Table 1 row. Growth: locality (smaller D) {loc_A} success A (pooled Spearman of success on D rho = "
         f"{rA_D.spearman_rho:.2f}, p {pf(rA_D.spearman_p)}; per P: {descA_D}); for replicator size, locality {loc_B}. "
         f"Diversity: dyadicity {dy_C} C (rho = {rC_P.spearman_rho:.2f}, p {pf(rC_P.spearman_p)}), and also {dy_A} A "
         f"and {dy_B} B. "
         + ("'Locality promotes growth' holds for success A only, not for replicator size; "
            if loc_A == "raises" and not (negB_D and not posB_D) else
            "'Locality promotes growth' holds for both A and B; " if loc_A == "raises" else
            "'Locality promotes growth' is not supported; ")
         + ("'dyadicity promotes diversity' holds." if dy_C == "raises" else "'dyadicity promotes diversity' does not hold.")),
        (217, "there was just one such anomaly case",
         f"Intended rule: {int((gi.category == 'runaway').sum())} runaway runs, {run_where} "
         f"({', '.join(f'P={r.P:.1f} seed {r.seed}' for r in gi[gi.runaway].itertuples()) or '-'}); "
         f"{int((gi.category == 'extinct').sum())} extinct; {int((gi.category == 'no_growth').sum())} no growth; "
         f"{int((gi.category == 'log_ge_300MB').sum())} with log >= 300 MB; operational thresholds as in "
         f"'Definitions used'. (10, 0.6) is {'also' if has_106 else 'not'} a runaway condition under the intended rule."),
        (221, "In each heatmap, the top right corner",
         "Still correct: Fig. 5 replacement keeps D increasing to the right and P = 0 on the top row."),
        (221, "Figure \\ref{fig:P-D-final}a shows that introducing spatial locality or dyadicity significantly increased",
         f"A(L,0) = {o.A:.0f}/30 [Wilson {o.A_wilson_lo:.1f}, {o.A_wilson_hi:.1f}]. Dyadicity {dy_A} A: mean over D by "
         "P " + ", ".join(f"P={P:.1f}: {s[0]:.1f}" for P, s in zip(P_VALUES, A_P))
         + f" (Cochran-Armitage p = {pf(rA_P.cochran_armitage_p)}; logistic P-block LR p = "
         f"{pf(jA.loc['P terms (P, P^2, DP)', 'p'])}). Locality {loc_A} A: mean over P by D "
         + ", ".join(f"D={D}: {s[0]:.1f}" for D, s in zip(D_VALUES, A_D))
         + f" (Spearman p = {pf(rA_D.spearman_p)}, Cochran-Armitage p = {pf(rA_D.cochran_armitage_p)}; logistic D-block "
         f"LR p = {pf(jA.loc['D terms (D, D^2, DP)', 'p'])}). Conditions with A significantly above the original "
         f"after Holm: {nA_hi}/59" + (" (30 runs per condition give little power per condition; cite the trends)."
                                      if nA_hi == 0 else ".")),
        (221, "Figure \\ref{fig:P-D-final}b shows that reducing the spatial interaction range",
         ("Supported" if negB_D and not posB_D else "Not supported" if not negB_D else "Mixed")
         + f" for the intended rule. Per-P Spearman of B vs D (the primary evidence): {descB_D}; i.e. locality "
         f"{loc_B}. Pooled over P (secondary; mixes strata with a {okB.max() / okB.min():.0f}-fold B variance range): "
         + (f"D = {D_VALUES[0]} vs D = 100 Cliff's delta {d10B.cliffs_delta.iloc[0]:+.2f}, Holm p "
            f"{pf(d10B.p_holm.iloc[0])}; " if len(d10B) else "")
         + f"other D vs D = 100 significant after Holm in {nDB}/{nDB_all}; B by D (mean over P) "
         + ", ".join(f"{D}: {s[0]:.2f}" for D, s in zip(D_VALUES, B_D))
         + f"; adjusted D slope {tP.loc[('D', 'B')].slope_adj:+.4f} per unit D, p (HC3) = "
         f"{pf(tP.loc[('D', 'B')].slope_p_hc3)}; quadratic-model D block p (HC3) = "
         f"{pf(jB.loc['D terms (D, D^2, DP)', 'p_hc3'])}. B vs original: {nB_hi} conditions higher, {nB_lo} lower "
         f"after Holm. Dyadicity {dy_B} B: " + ", ".join(f"P={P:.1f}: {s[0]:.2f}" for P, s in zip(P_VALUES, B_P))
         + f" (Spearman rho = {rB_P.spearman_rho:.2f}, p {pf(rB_P.spearman_p)}). "
         f"Largest B: {ck(bmax)} ({ti.B.max():.2f}), smallest {ck(ti.B.idxmin())} ({ti.B.min():.2f})."),
        (221, "Figure \\ref{fig:P-D-final}c shows that the cumulative number of observed patterns",
         ("Supported" if C_interior else "Not supported")
         + f": C {'increases monotonically' if C_mono else 'is not monotone'} in P (mean over D: "
         + ", ".join(f"P={P:.1f}: {s[0]:.0f}" for P, s in zip(P_VALUES, C_P))
         + f"), maximal at P = {P_Cmax:.1f}" + (" (an interior optimum)" if C_interior else " (no interior optimum)")
         + f"; Spearman rho = {rC_P.spearman_rho:.2f}, positive in {rC_P.strata_pos_sig}/{rC_P.n_strata} D strata. "
         f"Max C at {ck(cmax)} ({ti.C.max():.0f}). C above the original in {nC_hi}/59 conditions after Holm."),
        (221, "Finally, Figures \\ref{fig:P-D-final}d and \\ref{fig:3D-plot} highlighted",
         f"{'Still correct' if int(o.rank_ABC) > 1 else 'Not supported'}: original ranks {int(o.rank_ABC)}/60 by "
         f"A x B x C; best {ck(best)} at {b.ABC / o.ABC:.1f}x (Cliff's delta for C "
         f"{cbm.loc['C', 'cliffs_delta']:+.2f}, p Holm {pf(cbm.loc['C', 'p_holm'])})."),
        (223, "too frequent dyadic interaction would negatively affect",
         f"Size half: dyadicity {dy_B} B (rho = {rB_P.spearman_rho:.2f}) - "
         + ("consistent with the claim" if dy_B == "lowers" else "contradicts the claim")
         + f". Exploration half: C is maximal at P = {P_Cmax:.1f}"
         + (f" (P = {P_VALUES[-1]} vs P = 0 pooled: Cliff's delta {d_C10.cliffs_delta.iloc[0]:+.2f})" if len(d_C10) else "")
         + (" - consistent with a 'moderate dyadicity' optimum." if C_interior
            else " - contradicts the claim; the 'moderate dyadicity' optimum for C is not reproduced.")),
        (236, "moderate spatial locality ($D \\approx 0.7 L$) helped",
         f"Caption (b): locality {loc_B} (per-P Spearman of B vs D: {descB_D}); the largest B is at {ck(bmax)}; "
         f"colour range B {ti.B.min():.2f}-{ti.B.max():.2f}."),
        (236, "with $P \\approx 0.4$ being the best",
         f"Caption (c): C maximal at P = {P_Cmax:.1f} (largest condition {ck(cmax)}); colour range "
         f"{ti.C.min():.0f}-{ti.C.max():.0f}."),
        (243, "Markers are colored according to their proximity to the ideal outcome",
         f"New colour metric: A x B x C relative to the best condition (colourbar). Top 10 all at P >= "
         f"{top.index.get_level_values('P').min():.1f}. If `fig6_pairs` is used, state the jitter of A "
         f"(+/-{A_JITTER})."),
    ]
    # l.261 / l.264 / l.272 from the snapshot candidates
    o_rows = [dom_row(*ORIGINAL, t) for t in sorted(SNAP_TIMES)]
    o_rows = [r for r in o_rows if r is not None]
    o_sel = sel[(sel.D == ORIGINAL[0]) & (sel.P == ORIGINAL[1]) & (sel.n_candidates > 0)]
    if o_rows and len(o_sel):
        s0 = o_sel.iloc[0]
        k_dom = sum(int(r.n_dom_gt_half) for r in o_rows)
        n_dom = sum(int(r.n_candidates) for r in o_rows)
        hi_t = [str(int(r.t)) for r in o_rows if r.dom_median > 0.5]
        lo_t = [str(int(r.t)) for r in o_rows if r.dom_median <= 0.5]
        pan = dom_row(*ORIGINAL, int(s0.t))
        pan_med = pan.dom_median if pan is not None else s0.cand_dom_median
        typical = abs(s0.dominant_pattern_share - pan_med) <= 0.1
        oc = snapst[(snapst.D == ORIGINAL[0]) & (snapst.P == ORIGINAL[1]) & snapst.candidate]
        hi_s = oc[oc.dominant_pattern_share > 0.5].recent_rep_size
        lo_s = oc[oc.dominant_pattern_share <= 0.5].recent_rep_size
        recent_txt = (f"Low dominance goes with larger replicators: mean replication-event size over the 50 steps "
                      f"before the snapshot, median {lo_s.median():.1f} where share <= 0.5 (n = {len(lo_s)}) vs "
                      f"{hi_s.median():.1f} where share > 0.5 (n = {len(hi_s)}; `snapshot_stats.csv`, column "
                      f"recent_rep_size). " if len(hi_s) and len(lo_s) else "")
        t_prev, t_last = sorted(SNAP_TIMES)[-2], sorted(SNAP_TIMES)[-1]
        sp_, sl_ = oc[oc.t == t_prev].set_index("seed"), oc[oc.t == t_last].set_index("seed")
        drops = [q for q in sp_.index.intersection(sl_.index)
                 if sp_.loc[q, "dominant_pattern_share"] - sl_.loc[q, "dominant_pattern_share"] > 0.3]
        if drops:
            recent_txt += (f"Runs whose share fell by > 0.3 from t = {t_prev} to {t_last}: "
                           + ", ".join(f"seed {q} (share {sp_.loc[q, 'dominant_pattern_share']:.2f} -> "
                                       f"{sl_.loc[q, 'dominant_pattern_share']:.2f}, event size "
                                       f"{sp_.loc[q, 'recent_rep_size']:.1f} -> {sl_.loc[q, 'recent_rep_size']:.1f})"
                                       for q in drops) + ". ")
        txt261 = (f"Grown (L, 0) runs (successful, >= {SNAP_MIN_ACTIVE} cells), dominant-pattern share: "
                  + "; ".join(f"t = {int(r.t)}: median {r.dom_median:.2f} [{r.dom_min:.2f}, {r.dom_max:.2f}] over "
                              f"{int(r.n_candidates)} runs, {int(r.patterns_median)} distinct patterns" for r in o_rows)
                  + f". One pattern holds > 50% of the active cells in {k_dom}/{n_dom} (run, time) snapshots. The panel "
                  f"(seed {int(s0.seed)}, t = {int(s0.t)}) has share {s0.dominant_pattern_share:.2f} and "
                  f"{int(s0.n_patterns)} distinct patterns, {'close to' if typical else 'far from'} the candidate median "
                  f"at that time ({pan_med:.2f}). "
                  + (f"'One pattern dominating most of the time' fits t = {lstr(hi_t)} but not t = {lstr(lo_t)}. "
                     if hi_t and lo_t else
                     "'One pattern dominating' fits all snapshot times. " if hi_t else
                     "'One pattern dominating' fits none of the snapshot times. ")
                  + recent_txt
                  + "Reword the caption to describe the panel shown (and, if dominance over time is claimed, cite the "
                    "time-resolved shares); do not swap the seed to fit the old wording (the seed is fixed by the "
                    "stated rule).")
    else:
        txt261 = "Snapshots pending."
    entries.append((261, "The population tends to be homogeneous with only one", txt261))
    cmp_t = []
    for t in sorted(SNAP_TIMES):
        a, d = dom_row(*ORIGINAL, t), dom_row(20, 0.2, t)
        if a is not None and d is not None:
            more_var = d.dom_median < a.dom_median and d.patterns_median > a.patterns_median
            cmp_t.append((t, more_var, f"t = {t}: (20, 0.2) share {d.dom_median:.2f}, {d.patterns_median:.0f} "
                                        f"patterns vs (L, 0) {a.dom_median:.2f}, {a.patterns_median:.0f}"))
    yes_t = [str(t) for t, ok, _ in cmp_t if ok]
    no_t = [str(t) for t, ok, _ in cmp_t if not ok]
    entries.append((264, "While the population appears to be dominated by a major self-replicating pattern",
                    ("Candidate medians: " + "; ".join(s for *_, s in cmp_t) + ". 'More variation than (a)' (lower "
                     "dominant share and more distinct patterns) holds at t = " + (lstr(yes_t) or "none")
                     + " and not at t = " + (lstr(no_t) or "none") + ". " if cmp_t else "Snapshots pending. ")
                    + f"(20, 0.2) ranks {int(ti.loc[(20, 0.2)].rank_ABC)}/60 by A x B x C."))
    d_last = dom_row(20, 0.2, max(SNAP_TIMES))
    entries += [
        (272, "Finally, in the optimal setting with spatial locality and dyadic interactions",
         (f"(20, 0.2) is not optimal for the intended rule (rank {int(ti.loc[(20, 0.2)].rank_ABC)}/60); best by "
          f"A x B x C is {ck(best)}." if best != (20, 0.2) else "(20, 0.2) is also the best condition here.")
         + (f" Its grown runs at t = {max(SNAP_TIMES)}: dominant share median {d_last.dom_median:.2f}, "
            f"{d_last.patterns_median:.0f} distinct patterns." if d_last is not None else "")),
        (274, "Overall, the results indicate that spatial locality and dyadicity of interactions can have significant",
         f"'Significant impacts' holds, but the sentence should give directions. Dyadicity {dy_A} success A "
         f"(Cochran-Armitage p {pf(rA_P.cochran_armitage_p)}), {dy_B} replicator size B (p {pf(rB_P.spearman_p)}) and "
         f"{dy_C} exploration C (p {pf(rC_P.spearman_p)}). Locality {loc_A} A (p {pf(rA_D.spearman_p)}), {loc_B}, and "
         f"{loc_C}."),
        (300, "This mechanism also accounts for the isolated explosive-growth event",
         f"Intended rule: runaway at D = 10 in {r10}/{n10} runs vs {r20}/{n20} at D >= 20 (Fisher p = {pf(bp)}). "
         + ("A runaway also occurs at (10, 0.6) under the intended rule. " if has_106 else
            "No intended-rule run at (10, 0.6) is a runaway; the (10, 0.6) event cited in the text is from the "
            "Mathematica (typo) runs. ")
         + f"Intended-rule runaway runs are at {', '.join(ck((r.D, r.P)) for r in gi[gi.runaway].itertuples()) or '-'}."),
        (348, "spatial locality and dyadicity each had a positive effect",
         f"Locality (smaller D) {loc_A} A (mean over P {A_D[0][0]:.1f} at D={D_VALUES[0]} vs {A_D[-1][0]:.1f} at "
         f"D={D_VALUES[-1]}; per-P Spearman of success vs D: {descA_D}); it {loc_B}; it {loc_C} (C mean "
         f"over P: {C_D[0][0]:.0f} at D={D_VALUES[0]} vs {C_D[-1][0]:.0f} at D={D_VALUES[-1]}). Dyadicity {dy_A} A, "
         f"{dy_B} B and {dy_C} C (C maximal at P = {P_Cmax:.1f})."),
        (350, "Both locality and dyadicity have positive effects on the diversity",
         f"Locality {loc_C} (per-P Spearman of C vs D: {descC_D}; adjusted D slope on C p (HC3) = "
         f"{pf(tP.loc[('D', 'C')].slope_p_hc3)}, on log C p = {pf(tP.loc[('D', 'logC')].slope_p_hc3)}). Dyadicity "
         f"{dy_C} C (rho = {rC_P.spearman_rho:.2f}, positive in {rC_P.strata_pos_sig}/{rC_P.n_strata} D strata)."),
        (361, "we showed that spatial locality and dyadicity of interactions enrich",
         "See l.107 / l.137 / l.348 entries."),
    ]
    for ln, snip, txt in entries:
        at, note = locate(lines, ln, snip)
        w(f"- **l.{at}**{note} - replaces/qualifies: \"{snip}...\"  \n  {txt}")
    w("")
    w("## Data issues found\n")
    issues = []
    ext = {rule: set(map(tuple, runs[(runs.rule == rule) & (runs.category == "extinct")][["D", "P", "seed"]]
                         .itertuples(index=False))) for rule in ("intended", "as_coded")}
    if ext["intended"] or ext["as_coded"]:
        lst = lambda s: ", ".join(f"(D={d}, P={p:.1f}, seed {sd})" for d, p, sd in sorted(s))
        issues.append(f"Extinct runs, intended: {lst(ext['intended']) or 'none'}; as_coded: "
                      + ("the same (D, P, seed)" if ext["as_coded"] == ext["intended"] else lst(ext["as_coded"]) or "none")
                      + f". Runs with NaN `mean_rep_size` in the final 500 rows: {len(nan_runs)}, of which used: "
                      f"{int(nan_runs.used.sum())}, so the ensemble and per-run definitions of B "
                      + ("coincide." if int(nan_runs.used.sum()) == 0 else "may differ (see max difference above)."))
    ok = gi[gi.used]
    low = ok.nsmallest(3, "C_run")
    ol = ok[(ok.D == ORIGINAL[0]) & (ok.P == ORIGINAL[1])].nsmallest(2, "C_run")
    mm = ok.groupby("P").C_run.agg(["mean", "median"])
    issues.append("A run can pass Sayama's success criterion with very few patterns: smallest C among successful runs "
                  + ", ".join(f"{r.C_run:.0f} {ck((r.D, r.P))} seed {r.seed}" for r in low.itertuples())
                  + "; at (L, 0): " + ", ".join(f"seed {r.seed}: C = {r.C_run:.0f}, {r.n_active_final} cells at the end"
                                               for r in ol.itertuples())
                  + ". Mean/median of C over successful runs by P (pooled over D): "
                  + ", ".join(f"P={P:.1f}: {r['mean'] / r['median']:.2f}" for P, r in mm.iterrows())
                  + " (conditions.csv has both means and medians).")
    if n300 == 0 and len(rl):
        issues.append("No run reaches the 300 MB log threshold, so in the JAX data the log filter never excludes the "
                      "runaway runs; the runaway (event size) criterion is required to reproduce Sayama's exclusion.")
    if (sel.n_candidates == 0).any():
        issues.append("Snapshots pending for: " + ", ".join(sel[sel.n_candidates == 0].label) + ".")
    for s in issues:
        w(f"- {s}")
    (out / "summary.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
