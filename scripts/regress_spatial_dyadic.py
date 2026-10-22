#!/usr/bin/env python3
"""Quadratic response-surface regression of the spatial-dyadic SCHC measures (npj R1.1).

Model (Sayama, e-mail of 2026-09-26):  Y = a + b D + c D^2 + d P + e P^2 + f D P + error
fitted to three data sets:
  mathematica : Sayama's Mathematica runs (swap typo), review/Sayama-addition/SD-SCHC-experimental-data.csv
  as_coded    : JAX, dyadic rule as coded (reproduces the typo)
  intended    : JAX, dyadic rule as intended
Y = A (successful runs per condition, 60 points), B (mean active cells per replication event in
the final 500 steps, per successful run) and C (cumulative pattern types at t=2000, per successful
run). OLS with classical standard errors (as Mathematica's NonlinearModelFit) and HC3 robust ones.
A is also fitted as a binomial GLM (logit link) on the 0/1 success of every run.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SAYAMA_CSV = ROOT / "tex/Large-Hash-Chemistry/review/Sayama-addition/SD-SCHC-experimental-data.csv"
SD_ROOT = ROOT / "results" / "spatial_dyadic"
TERMS = ["1", "D", "D^2", "P", "P^2", "DP"]

spec = importlib.util.spec_from_file_location("asd", ROOT / "scripts" / "analyze_spatial_dyadic.py")
asd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asd)


def design(D, P):
    D, P = np.asarray(D, float), np.asarray(P, float)
    return np.column_stack([np.ones_like(D), D, D ** 2, P, P ** 2, D * P])


def ols(X, y):
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    r = y - X @ beta
    se = np.sqrt(np.diag(XtXi) * (r @ r) / (n - k))
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    se_hc3 = np.sqrt(np.diag(XtXi @ (X.T * (r / (1 - h)) ** 2) @ X @ XtXi))
    p = 2 * stats.t.sf(np.abs(beta / se), n - k)
    p_hc3 = 2 * stats.t.sf(np.abs(beta / se_hc3), n - k)
    return beta, se, p, p_hc3, n


def logit(X, y, iters=100):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        step = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = 1 / (1 + np.exp(-(X @ beta)))
    cov = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    se = np.sqrt(np.diag(cov))
    return beta, se, 2 * stats.norm.sf(np.abs(beta / se)), len(y)


def load_mathematica():
    d = pd.read_csv(SAYAMA_CSV, header=None, names=["D", "P", "measure", "value"])
    d["P"] = d["P"].round(1)
    d["m"] = d["measure"].str[0]
    return {m: d[d["m"] == m][["D", "P", "value"]].reset_index(drop=True) for m in "ABC"}


def load_jax(rule):
    A, B, C, bern = [], [], [], []
    for D in asd.D_VALUES:
        for P in asd.P_VALUES:
            sub = "intended" if P == 0.0 else rule  # P=0 does not depend on the rule
            runs = asd.load_condition(SD_ROOT / sub / f"D{D}_P{P:.1f}")
            if runs is None:
                raise FileNotFoundError(f"{sub}/D{D}_P{P:.1f}")
            ok = [r for r in runs if r[0] and not r[3]]
            A.append((D, P, len(ok)))
            bern += [(D, P, float(r[0] and not r[3])) for r in runs]
            B += [(D, P, float(np.nanmean(r[1]))) for r in ok]
            C += [(D, P, r[2]) for r in ok]
    f = lambda rows: pd.DataFrame(rows, columns=["D", "P", "value"])
    return {"A": f(A), "B": f(B), "C": f(C)}, f(bern)


def fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def main():
    sets = {"mathematica": (load_mathematica(), None)}
    for rule in ("as_coded", "intended"):
        sets[rule] = load_jax(rule)
    rows = []
    for name, (data, bern) in sets.items():
        for m in "ABC":
            df = data[m]
            beta, se, p, p3, n = ols(design(df.D, df.P), df.value.to_numpy(float))
            for t, b, s, pp, pp3 in zip(TERMS, beta, se, p, p3):
                rows.append(dict(data=name, measure=m, model="OLS", n=n, term=t, coef=b, se=s,
                                 p=pp, p_hc3=pp3))
        if bern is not None:
            beta, se, p, n = logit(design(bern.D, bern.P), bern.value.to_numpy(float))
            for t, b, s, pp in zip(TERMS, beta, se, p):
                rows.append(dict(data=name, measure="A", model="logit", n=n, term=t, coef=b, se=s,
                                 p=pp, p_hc3=np.nan))
    out = pd.DataFrame(rows)
    OUT = SD_ROOT / "analysis"
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "regression_quadratic.csv", index=False)

    # vertex of C in P (dC/dP = d + 2 e P + f D = 0) at the mean D, if concave
    for name in sets:
        c = out[(out.data == name) & (out.measure == "C") & (out.model == "OLS")].set_index("term").coef
        Dm = 55.0
        vert = -(c["P"] + c["DP"] * Dm) / (2 * c["P^2"]) if c["P^2"] < 0 else np.nan
        print(f"[{name}] C: P^2 coef {c['P^2']:.0f} -> "
              + (f"maximum at P = {vert:.2f} (D=55)" if np.isfinite(vert) else "no interior maximum (convex/monotone)"))
    pd.set_option("display.width", 200)
    show = out.assign(coef=out.coef.map(lambda v: f"{v:.4g}"), p=out.p.map(fmt_p),
                      p_hc3=out.p_hc3.map(lambda v: "" if np.isnan(v) else fmt_p(v)))
    for (name, m, model), g in show.groupby(["data", "measure", "model"], sort=False):
        print(f"\n== {name} | {m} | {model} (n={g.n.iloc[0]})")
        print(g[["term", "coef", "p", "p_hc3"]].to_string(index=False))


if __name__ == "__main__":
    main()
