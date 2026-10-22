#!/usr/bin/env python3
"""
Statistics for npj Extension I (spatial-dyadic SCHC, L=100) on three data sets:
  mathematica  Sayama's per-run data (SD-SCHC-experimental-data.csv: A per condition,
               B and C per successful run), produced by the Mathematica code in which the
               dyadic order swap never runs;
  as_coded     JAX runs reproducing that behaviour (scripts/run_spatial_dyadic_sweep.py);
  intended     JAX runs with the dyadic rule as described in the papers.

1. Per-condition agreement of the per-run B and C distributions, Mathematica vs JAX as_coded
   (Mann-Whitney U, Holm-corrected across conditions).
2. Sayama's quadratic model  y = a + b D + c D^2 + d P + e P^2 + f D P  by OLS, with y =
   A (per condition, as in his analysis), B and C (per successful run); and the same model as a
   logistic regression on per-run success (A as a binary variable).
Runaway runs (a replication event >= 1000 cells) count as failures, as in the original analysis
where the explosive run (380 MB log) failed the 300 MB filter.

Output: results/spatial_dyadic/stats/{regression.csv,agreement.csv,report.md}
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
D_VALUES = list(range(10, 101, 10))
P_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TERMS = ["a (1)", "b (D)", "c (D^2)", "d (P)", "e (P^2)", "f (D P)"]
RUNAWAY_SIZE = 1000


def design(D, P):
    D, P = np.asarray(D, float), np.asarray(P, float)
    return np.column_stack([np.ones_like(D), D, D**2, P, P**2, D * P])


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    return beta, se, 2 * stats.t.sf(np.abs(t), dof)


def logistic(X, y, iters=100):
    """Newton-Raphson logistic regression on standardized columns; Wald tests."""
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0)
    Z = np.column_stack([X[:, 0], (X[:, 1:] - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, Z.T @ (y - p))
        w += step
        if np.max(np.abs(step)) < 1e-10:
            break
    cov = np.linalg.inv(H)
    se_z = np.sqrt(np.diag(cov))
    pval = 2 * stats.norm.sf(np.abs(w / se_z))
    # back to the original scale (coefficients only; p-values are scale invariant for slopes)
    beta = np.concatenate([[w[0] - np.sum(w[1:] * mu / sd)], w[1:] / sd])
    return beta, pval


def load_mathematica(path):
    A, B, C = {}, defaultdict(list), defaultdict(list)
    for D, P, name, v in csv.reader(open(path)):
        key = (int(D), round(float(P), 1))
        if name.startswith("A"):
            A[key] = int(float(v))
        elif name.startswith("B"):
            B[key].append(float(v))
        else:
            C[key].append(float(v))
    return A, B, C


def load_jax(root, rule):
    A, B, C, runs = {}, defaultdict(list), defaultdict(list), {}
    for D in D_VALUES:
        for P in P_VALUES:
            d = root / ("intended" if P == 0.0 else rule) / f"D{D}_P{P:.1f}"
            ok_flags = []
            for f in sorted(d.glob("seed_*.csv")):
                x = np.genfromtxt(f, delimiter=",", names=True)
                ok = (1e6 < x["log_bytes_cum"][-1] < 3e8 and x["n_active"][-1] > 0 and len(x) == 2000
                      and np.nanmax(x["max_rep_size"]) < RUNAWAY_SIZE)
                ok_flags.append(ok)
                if ok:
                    B[(D, P)].append(float(np.nanmean(x["mean_rep_size"][-500:])))
                    C[(D, P)].append(float(x["cum_pattern_types"][-1]))
            A[(D, P)] = int(sum(ok_flags))
            runs[(D, P)] = len(ok_flags)
    return A, B, C, runs


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sayama-csv", default=str(ROOT / "tex/Large-Hash-Chemistry/review/Sayama-addition/SD-SCHC-experimental-data.csv"))
    ap.add_argument("--root", default=str(ROOT / "results" / "spatial_dyadic"))
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "stats"
    out.mkdir(parents=True, exist_ok=True)

    data = {"mathematica": load_mathematica(args.sayama_csv) + ({k: 30 for k in [(D, P) for D in D_VALUES for P in P_VALUES]},)}
    for rule in ("as_coded", "intended"):
        data[rule] = load_jax(root, rule)

    md = ["# Extension I statistics: Mathematica vs JAX (as coded / as intended)\n"]

    # 1. agreement of per-run distributions, Mathematica vs JAX as_coded
    rows, pB, pC = [], [], []
    mA, mB, mC, _ = data["mathematica"]
    jA, jB, jC, _ = data["as_coded"]
    keys = [(D, P) for P in P_VALUES for D in D_VALUES]
    for k in keys:
        b = stats.mannwhitneyu(mB[k], jB[k]).pvalue if len(mB[k]) > 1 and len(jB[k]) > 1 else np.nan
        c = stats.mannwhitneyu(mC[k], jC[k]).pvalue if len(mC[k]) > 1 and len(jC[k]) > 1 else np.nan
        a = stats.fisher_exact([[mA[k], 30 - mA[k]], [jA[k], 30 - jA[k]]])[1]
        rows.append((k, a, b, c)); pB.append(b); pC.append(c)
    aA = holm([r[1] for r in rows]); aB = holm(pB); aC = holm(pC)
    with open(out / "agreement.csv", "w") as f:
        f.write("D,P,A_math,A_jax,fisher_p,B_mwu_p,C_mwu_p,fisher_p_holm,B_p_holm,C_p_holm\n")
        for (k, a, b, c), ha, hb, hc in zip(rows, aA, aB, aC):
            f.write(f"{k[0]},{k[1]},{mA[k]},{jA[k]},{a:.4g},{b:.4g},{c:.4g},{ha:.4g},{hb:.4g},{hc:.4g}\n")
    raw = lambda ps: sum(p < 0.05 for p in ps)
    md.append("## 1. Mathematica vs JAX as-coded, per condition (60 conditions)\n")
    md.append(f"- A (Fisher exact on success counts): p<0.05 in {raw([r[1] for r in rows])}/60, after Holm {sum(aA < 0.05)}/60")
    md.append(f"- B (Mann-Whitney on per-run values): p<0.05 in {raw(pB)}/60 (expected ~3 by chance), after Holm {sum(aB < 0.05)}/60")
    md.append(f"- C (Mann-Whitney on per-run values): p<0.05 in {raw(pC)}/60, after Holm {sum(aC < 0.05)}/60\n")

    # 2. regressions
    reg_lines = ["data,response,method,term,estimate,p_value"]
    for resp in ("A", "B", "C", "A_logistic"):
        md.append(f"## 2. {resp}: y = a + b D + c D^2 + d P + e P^2 + f D P\n")
        md.append("| term | " + " | ".join(f"{s} estimate (p)" for s in data) + " |")
        md.append("|---|" + "---|" * len(data))
        cols = {}
        for src, (A, B, C, n) in data.items():
            if resp == "A":
                X = design([k[0] for k in keys], [k[1] for k in keys]); y = np.array([A[k] for k in keys], float)
                beta, _, p = ols(X, y); method = "OLS per condition"
            elif resp == "A_logistic":
                Dl, Pl, yl = [], [], []
                for k in keys:
                    Dl += [k[0]] * n[k]; Pl += [k[1]] * n[k]; yl += [1] * A[k] + [0] * (n[k] - A[k])
                beta, p = logistic(design(Dl, Pl), np.array(yl, float)); method = "logistic per run"
            else:
                V = B if resp == "B" else C
                Dl = [k[0] for k in keys for _ in V[k]]; Pl = [k[1] for k in keys for _ in V[k]]
                y = np.array([v for k in keys for v in V[k]])
                beta, _, p = ols(design(Dl, Pl), y); method = "OLS per run"
            cols[src] = (beta, p)
            for t, b_, p_ in zip(TERMS, beta, p):
                reg_lines.append(f"{src},{resp},{method},{t},{b_:.6g},{p_:.4g}")
        for i, t in enumerate(TERMS):
            cells = []
            for src in data:
                b_, p_ = cols[src][0][i], cols[src][1][i]
                star = "***" if p_ < 0.001 else "**" if p_ < 0.01 else "*" if p_ < 0.05 else ""
                cells.append(f"{b_:.4g} ({p_:.2g}){star}")
            md.append(f"| {t} | " + " | ".join(cells) + " |")
        md.append("")
    (out / "regression.csv").write_text("\n".join(reg_lines) + "\n")
    (out / "report.md").write_text("\n".join(md) + "\n")
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
