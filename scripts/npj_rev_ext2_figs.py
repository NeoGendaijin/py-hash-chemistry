#!/usr/bin/env python3
"""Extension II (Sec. 4) figures for the npj Complexity revision, corrected model.

Redraws the two SCHC case-study figures of the manuscript from the CPU-engine runs of the
corrected model (winner_rule='original', score='mix'). It adapts
scripts/generate_fig_schc_figs.py (left unchanged) to the new data layout and drops the
periodic-boundary control.

fig_schc_phenomenon
  (a) L=200, seed 0: snapshots at t=5,000 and 10,000 plus a 60x60 zoom of the t=10,000 state
      centred on the largest component; results/cpu_snapshots/t{T}/... Cells are coloured by
      type; the largest component is outlined in black (cell-edge outline) whenever it covers
      less than --outline-max-fill (0.05) of the grid.
  (b) L=400, seed 5: snapshots at t=2,000 / 5,000 / 10,000. A largest component above that
      fill (the sparse runaway network) is not outlined; it keeps its type colours and all
      other occupied cells are drawn light grey, so the extent of the giant is visible.
  (c) Final mean component size per run vs L (log), with the ensemble median. Sizes on the
      coarse grid (L=200..400, step 20; --coarse-L) are blue circles, the onset-scan sizes
      (--onset-L: 120-180, 210, 230, 250) orange diamonds and the original SCHC size L=100
      (--ref-L) a blue triangle. The shaded transition window comes from the data (--window).
  (d) Runaway fraction vs L with Wilson 95% intervals and the fitted logistic curve; the
      published model re-run with the same engine (results/cpu_scan/legacy_xor) in grey.
      Sizes with runs still missing are drawn with open markers (see the footnote).
  (e) Long-run novelty at L=200 (median over the runs of results/cpu_novelty, metrics every
      step): cumulative pattern types (top) and mean component size (bottom).

fig_schc_mechanism
  (a) Fill fraction s_max/L^2 of the dominant component at t=20,000, runaway runs only.
  (b) Pearson r between the ensemble-mean size and score trajectories (steps 2,000-10,000)
      with 95% bootstrap intervals over runs, identical to Table 2: rows of
      results/npj_rev/ext2_tables/table2.csv whose size_status is 'complete' are read as they
      are; every other L is computed here with the tables script's own loader and table2_row()
      (score NaN after extinction and averaged over live runs, bootstrap weights from
      default_rng([SEED, L]), 10,000 resamples). L whose 100 trajectories are not all complete
      are provisional (open markers, footnote). Second series: per-run r, back-transformed
      mean Fisher z with its bootstrap interval (Table 2 column r_run).
  (c,d) (L, mu) and (L, p_death) diagrams from results/cpu_sensitivity: colour = runaway
      fraction at t=10,000, annotation = median final mean component size. The mu=0.002 row
      and the p_death=0.001 row are the same baseline run set (original_mix: mu=0.002/0.999,
      p_death=0.001); the exact-mu runs original_mix_mu0.002 are used for the mu row only
      while the baseline is missing (footnote).
  (e) Nucleation kinetics: fraction of runs whose mean component size has exceeded 100 by
      time t (first passage), for L across the onset region and the plateau (labelled at the
      line ends).

Rules (paper): runaway = final mean component size > 100 at the run's last row; extinct runs
(n_active == 0) are non-runaway with size 0 and are padded with zeros after extinction.

Incomplete data. The scan without early stop (results/cpu_scan_full) is being filled. Runs of
the early-stopped scan (results/cpu_scan/original_mix) that never reached the stop threshold
have been copied into it; the stopped runs are being re-run. Until a stopped run's re-run
exists, it is counted as runaway (its trajectory is identical up to the stop, so the
classification and the first-passage time are known) and drawn with an open symbol at its
early-stop value in (c) and mechanism (a). It is left out of everything that needs the
trajectory after the stop (median in (c) excepted, see the report). --no-provisional uses
complete runs only. Sizes that still miss runs (no file in either scan yet, e.g. the onset
scan while it runs) are listed in the footnote: the unfinished runs are mostly the slower
runaway ones, so (c)-(e) and mechanism (a,e) are biased low there. A footnote on the figure
lists what is provisional or pending; it disappears once all data are in. Panels without
data show a 'pending' placeholder.

Outputs (results/npj_rev/ext2fig/): fig_schc_phenomenon.{pdf,png},
fig_schc_mechanism.{pdf,png}, runs.csv, by_L.csv, window_fit.csv, sensitivity.csv,
nucleation.csv, novelty.csv, report.md.

Usage:
  .venv/bin/python scripts/npj_rev_ext2_figs.py                      # everything
  .venv/bin/python scripts/npj_rev_ext2_figs.py --only phenomenon --window 200 260
  .venv/bin/python scripts/npj_rev_ext2_figs.py --no-provisional --nuc-L 200 240 300 400
  .venv/bin/python scripts/npj_rev_ext2_figs.py --r-source compute   # (b) without table2.csv
"""
from __future__ import annotations

import os

# keep BLAS single-threaded: simulations are running on the other cores
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, hsv_to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import ndimage, optimize

ROOT = Path(__file__).resolve().parents[1]
# Table 2 conventions (size-score correlation) come from the tables script itself. Imported
# before the theme below, which therefore overrides its rcParams.
sys.path.insert(0, str(ROOT / "scripts"))
import npj_rev_ext2_tables as ext2t  # noqa: E402

RUN = 100.0          # runaway threshold on the final mean component size
STOP = 2000.0        # early-stop threshold of results/cpu_scan
T_END = 20000        # scan horizon
T_SENS = 10000       # sensitivity horizon
BOOT_SEED = 20260925

# ---------------- seaborn theme (as in generate_fig_schc_figs.py) ----------------
sns.set_theme(context="paper", style="ticks", font="DejaVu Sans")
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a", "figure.dpi": 300, "savefig.dpi": 300,
})
PAL = sns.color_palette("deep")
C_MAIN, C_ADD, C_MED = PAL[0], PAL[1], "#2b2b2b"   # blue, orange (colour-blind safe pair)
C_LEG = "#a3a3a3"                                  # published model, muted
C_PAT = PAL[4]                                     # novelty panel
BAND = "#e9e6ef"                                   # transition-window shade
MM = 1 / 25.4
DOUBLE_COL = 180 * MM
GOLDEN = 0.618033988749895


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def short(p: Path, n: int = 44) -> str:
    """rel(p), shortened to its last components when long (pending placeholders)."""
    s = rel(p)
    if len(s) <= n:
        return s
    parts = Path(s).parts
    for k in range(len(parts) - 1, 0, -1):
        t = ".../" + "/".join(parts[-k:])
        if len(t) <= n:
            return t
    return ".../" + parts[-1]


def tex_int(v) -> str:
    """Integer with thousands separators for use inside $...$ ('{,}' avoids the math-mode gap)."""
    return f"{int(v):,}".replace(",", "{,}")


# ---------------- marker groups (L=100 reference, Delta L=20 grid, onset scan) ----------------
C_REF = PAL[0]
GROUPS = {   # group -> (colour, marker, marker size in points, legend label)
    "grid": (C_MAIN, "o", 3.6, "Coarse scan"),
    "onset": (C_ADD, "D", 3.2, "Onset scan"),
    "ref": (C_REF, "^", 3.9, r"$L{=}100$"),
}


def group_of(L: int, a) -> str:
    if L in a.ref_L:
        return "ref"
    if L in a.coarse_L:
        return "grid"
    return "onset"            # --onset-L, and any other size (warned about in main)


def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = map(int, part.split("-"))
            out += list(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def list_runs(d: Path) -> dict[int, Path]:
    rd = d / "runs"
    if not rd.is_dir():
        return {}
    out = {}
    for f in rd.iterdir():
        m = re.fullmatch(r"seed_(\d+)\.csv", f.name)
        if m:
            out[int(m.group(1))] = f
    return out


def list_L(root: Path) -> list[int]:
    if not root.is_dir():
        return []
    return sorted(int(d.name[1:]) for d in root.iterdir() if re.fullmatch(r"L\d+", d.name) and d.is_dir())


def on_grid(step: np.ndarray, vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Value at each grid step: the last sample at or before it (forward fill after the run ends)."""
    i = np.searchsorted(step, grid, side="right") - 1
    out = vals[np.clip(i, 0, None)].astype(float)
    out[i < 0] = np.nan
    return out


# ======================= data =======================
def summarize(f: Path, L: int, t_end: int):
    x = pd.read_csv(f)
    step = x["step"].to_numpy()
    ms = x["mean_size"].to_numpy(float)
    last = x.iloc[-1]
    extinct = int(last["n_active"]) == 0
    complete = extinct or int(last["step"]) >= t_end
    stopped = (not complete) and float(last["mean_size"]) >= STOP
    over = np.flatnonzero(ms > RUN)
    first = float(step[over[0]]) if len(over) else np.inf
    rec = dict(L=L, last_step=int(last["step"]), final_mean=float(last["mean_size"]),
               final_max=float(last["max_size"]), fill=float(last["max_size"]) / L ** 2,
               extinct=extinct, complete=complete, stopped=stopped, first_cross=first,
               ever_runaway=bool(len(over)),
               fell_back=bool(len(over)) and complete and float(last["mean_size"]) <= RUN)
    return rec


def load_scan(full_root: Path, stop_root: Path, seeds: list[int], t_end: int = T_END) -> pd.DataFrame:
    """Merge the no-stop scan with the early-stopped one (see module docstring)."""
    recs = []
    for L in sorted(set(list_L(full_root)) | set(list_L(stop_root))):
        fr, sr = list_runs(full_root / f"L{L}"), list_runs(stop_root / f"L{L}")
        for s in seeds:
            if s in fr:
                rec = summarize(fr[s], L, t_end)
                rec["source"] = "full" if rec["complete"] else "full-incomplete"
            elif s in sr:
                rec = summarize(sr[s], L, t_end)
                rec["source"] = ("stopscan-complete" if rec["complete"]
                                 else "stopped" if rec["stopped"] else "stopscan-incomplete")
            else:
                continue
            rec["seed"] = s
            recs.append(rec)
    runs = pd.DataFrame(recs)
    if len(runs):
        runs["runaway"] = (runs["complete"] & (runs["final_mean"] > RUN)) | runs["stopped"]
    return runs


def load_simple(root: Path, seeds: list[int] | None, t_end: int, Ls: list[int] | None = None) -> pd.DataFrame:
    """Final-state table for a root of complete runs (legacy re-run, sensitivity)."""
    recs = []
    for L in list_L(root):
        if Ls is not None and L not in Ls:
            continue
        for s, f in sorted(list_runs(root / f"L{L}").items()):
            if seeds is not None and s not in seeds:
                continue
            rec = summarize(f, L, t_end)
            rec["seed"] = s
            recs.append(rec)
    df = pd.DataFrame(recs)
    if len(df):
        # the legacy re-run stops at mean size 2000: such runs are runaway
        df["runaway"] = (df["final_mean"] > RUN) | df["stopped"]
        df["usable"] = df["complete"] | df["stopped"]
    return df


def by_L_table(runs: pd.DataFrame, seeds: list[int], provisional: bool) -> pd.DataFrame:
    rows = []
    for L, g in runs.groupby("L"):
        comp = g[g["complete"]]
        pend = g[g["stopped"]]
        use = pd.concat([comp, pend]) if provisional else comp
        k, n = int(use["runaway"].sum()), len(use)
        lo, hi = wilson(k, n)
        rw = comp[comp["runaway"]]
        rows.append(dict(
            L=L, n_expected=len(seeds), n_complete=len(comp), n_stopped_pending=len(pend),
            n_missing=len(seeds) - len(comp) - len(pend), n_used=n, partial=n < len(seeds),
            runaway_k=k,
            runaway_frac=k / n if n else np.nan, runaway_lo=lo, runaway_hi=hi,
            n_extinct=int(comp["extinct"].sum()), n_fell_back=int(comp["fell_back"].sum()),
            median_final_mean=float(np.median(use["final_mean"])) if n else np.nan,
            mean_final_mean_complete=float(comp["final_mean"].mean()) if len(comp) else np.nan,
            n_runaway_complete=len(rw),
            fill_median_runaway_t20k=float(rw["fill"].median()) if len(rw) else np.nan,
            fill_median_runaway_with_pending=(float(pd.concat([rw, pend])["fill"].median())
                                              if len(rw) + len(pend) else np.nan),
        ))
    return pd.DataFrame(rows)


# ---------------- transition window: logistic fit to the runaway fraction ----------------
def _logistic(L, pmax, L50, w):
    return pmax / (1 + np.exp(-(L - L50) / w))


def fit_logistic(L: np.ndarray, k: np.ndarray, n: np.ndarray, start=None):
    """Binomial ML fit of p(L) = pmax / (1 + exp(-(L - L50)/w)); returns (pmax, L50, w) or None.

    Multi-start Nelder-Mead; `start` = (pmax, L50, w) restricts it to one start (bootstrap refits)."""
    L, k, n = map(lambda a: np.asarray(a, float), (L, k, n))
    if k.sum() == 0 or len(L) < 3:
        return None

    def nll(th):
        pmax = 1 / (1 + np.exp(-th[0]))
        p = np.clip(_logistic(L, pmax, th[1], np.exp(th[2])), 1e-12, 1 - 1e-12)
        return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum()

    if start is not None:
        pm = np.clip(start[0], 0.02, 0.98)
        starts = [[np.log(pm / (1 - pm)), start[1], np.log(max(start[2], 1e-2))]]
    else:
        pm0 = np.clip(k[L >= np.median(L)].sum() / max(n[L >= np.median(L)].sum(), 1), 0.02, 0.98)
        starts = [[np.log(pm0 / (1 - pm0)), L50, lw] for L50 in np.linspace(L.min(), L.max(), 7)
                  for lw in (np.log(5.0), np.log(20.0), np.log(60.0))]
    best = None
    for x0 in starts:
        r = optimize.minimize(nll, x0, method="Nelder-Mead",
                              options=dict(xatol=1e-4, fatol=1e-7, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    pmax, L50, w = 1 / (1 + np.exp(-best.x[0])), best.x[1], np.exp(best.x[2])
    return float(pmax), float(L50), float(np.clip(w, 1e-3, 1e4))


def transition_window(tab: pd.DataFrame, spec, n_boot: int = 300):
    """Window to shade: spec 'auto' (10-90% rise of the logistic fit), 'none', or (lo, hi)."""
    info = dict(method=str(spec), pmax=np.nan, L50=np.nan, w=np.nan, lo=np.nan, hi=np.nan,
                L50_lo=np.nan, L50_hi=np.nan, win_lo_lo=np.nan, win_lo_hi=np.nan,
                win_hi_lo=np.nan, win_hi_hi=np.nan)
    t = tab[tab["n_used"] > 0]
    fit = fit_logistic(t["L"], t["runaway_k"], t["n_used"]) if len(t) else None
    if fit is not None:
        pmax, L50, w = fit
        info.update(pmax=pmax, L50=L50, w=w)
        rng = np.random.default_rng(BOOT_SEED)
        bs = []
        for _ in range(n_boot):
            kb = rng.binomial(t["n_used"].to_numpy(), (t["runaway_k"] / t["n_used"]).to_numpy())
            fb = fit_logistic(t["L"], kb, t["n_used"], start=fit)
            if fb is not None:
                bs.append((fb[1], fb[1] - fb[2] * np.log(9), fb[1] + fb[2] * np.log(9)))
        if bs:
            bs = np.array(bs)
            q = np.percentile(bs, [2.5, 97.5], axis=0)
            info.update(L50_lo=q[0, 0], L50_hi=q[1, 0], win_lo_lo=q[0, 1], win_lo_hi=q[1, 1],
                        win_hi_lo=q[0, 2], win_hi_hi=q[1, 2])
    if isinstance(spec, tuple):
        info.update(lo=float(spec[0]), hi=float(spec[1]))
    elif spec == "auto" and fit is not None:
        info.update(lo=max(float(t["L"].min()), info["L50"] - info["w"] * np.log(9)),
                    hi=min(float(t["L"].max()), info["L50"] + info["w"] * np.log(9)))
    return info


# ---------------- size-score correlation ----------------
R_COLS = ["r", "r_lo", "r_hi", "r_run", "r_run_lo", "r_run_hi", "r_run_median"]


def correlations(Ls, a) -> tuple[pd.DataFrame, str]:
    """Size-score correlation per L with the Table 2 conventions (see the module docstring).

    Rows of table2.csv with size_status 'complete' are taken as they are (--r-source auto);
    every other L is computed with npj_rev_ext2_tables.load_main() and table2_row(), i.e. with
    the same data selection, extinct-run handling (score NaN after extinction, averaged over
    the live runs), bootstrap weights (default_rng([SEED, L]), 10,000 resamples) and per-run r
    (live samples only) as the tables script. table2_row() is called with allow_partial, so an
    L with at least 5 but fewer than all trajectories gets provisional values (r_status)."""
    have: dict[int, dict] = {}
    if a.r_source == "auto" and a.table2.exists():
        t2 = pd.read_csv(a.table2)
        need = {"L", "size_status", "n_nostop", *R_COLS}
        if need <= set(t2.columns):
            for _, r in t2.iterrows():
                if r["size_status"] == "complete" and np.isfinite(r["r"]):
                    have[int(r["L"])] = dict(L=int(r["L"]), n_r=int(r["n_nostop"]),
                                             **{c: float(r[c]) for c in R_COLS},
                                             r_status="complete", r_source=rel(a.table2))
        else:
            print(f"warning: {rel(a.table2)} lacks columns {sorted(need - set(t2.columns))}; computing all L")
    # point the tables script's loader at this script's inputs
    ext2t.FULL, ext2t.STOPPED, ext2t.SEEDS = a.full_root, a.stop_root, list(a.seeds)
    rows = []
    for L in sorted(int(v) for v in Ls):
        if L in have:
            rows.append(have[L])
            continue
        recs, trajs = ext2t.load_main(L)
        df = pd.DataFrame(recs)
        n_known = int(df["status"].isin(ext2t.KNOWN).sum())
        row = ext2t.table2_row(L, df, trajs, ext2t.tu.B, True, [])
        del trajs
        if not np.isfinite(row["r"]):
            continue
        rows.append(dict(L=L, n_r=n_known, **{c: float(row[c]) for c in R_COLS},
                         r_status="complete" if row["size_status"] == "complete" else "provisional",
                         r_source="computed here (npj_rev_ext2_tables.table2_row)"))
    tab = pd.DataFrame(rows, columns=["L", "n_r", *R_COLS, "r_status", "r_source"])
    src = sorted(set(tab["r_source"])) if len(tab) else []
    used = "; ".join(f"{s}: L=" + ",".join(str(L) for L in tab.loc[tab["r_source"] == s, "L"]) for s in src)
    return tab, used or "none"


# ---------------- novelty, sensitivity, snapshots ----------------
def load_novelty(root: Path, t_end: int = T_END):
    files = list_runs(root)
    if not files:
        return None
    grid = None
    pats, sizes = [], []
    for s in sorted(files)[:40]:
        x = pd.read_csv(files[s], usecols=["step", "mean_size", "cum_pattern_types", "n_active"])
        last = x.iloc[-1]
        if int(last["step"]) < t_end and int(last["n_active"]) != 0:
            continue          # incomplete run (should not happen: files are written at the end)
        if grid is None:
            si = int(np.median(np.diff(x["step"].to_numpy()[:50]))) or 1
            grid = np.arange(0, t_end + 1, si)
        st = x["step"].to_numpy()
        pats.append(on_grid(st, x["cum_pattern_types"].to_numpy(float), grid))
        sizes.append(on_grid(st, x["mean_size"].to_numpy(float), grid))
    if not pats:
        return None
    P, S = np.stack(pats), np.stack(sizes)
    q = lambda A, p: np.nanpercentile(A, p, axis=0)
    return pd.DataFrame(dict(step=grid, n=len(P), pat_median=q(P, 50), pat_q25=q(P, 25), pat_q75=q(P, 75),
                             size_median=q(S, 50), size_q25=q(S, 25), size_q75=q(S, 75)))


SENS_MU = [0.001, 0.002, 0.005, 0.01]
SENS_PD = [0.0005, 0.001, 0.005, 0.01]


def sens_dirs(root: Path, kind: str, val: float) -> list[Path]:
    """Directories holding one sensitivity condition, in order of preference.

    The baseline run set original_mix (mu = 0.002/0.999, p_death = 0.001; J4 writes it for
    --death-prob 0.001) fills both the mu = 0.002 row and the p_death = 0.001 row, so the shared
    cell shows the same runs in (c) and (d). J4's --mu 0.002 runs (original_mix_mu0.002, mu
    exactly 0.002) are a fallback for the mu row while the baseline is missing."""
    base = root / "original_mix"
    if kind == "mu":
        cands = ([base, root / "original_mix_mu0.002"] if np.isclose(val, 0.002)
                 else [root / f"original_mix_mu{val:g}"])
    else:
        cands = [base] if np.isclose(val, 0.001) else [root / f"original_mix_pd{val:g}"]
    return [d for d in cands if d.is_dir()]


def load_sensitivity(root: Path, Ls: list[int], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for kind, vals in (("mu", SENS_MU), ("p_death", SENS_PD)):
        for v in vals:
            dirs = sens_dirs(root, "mu" if kind == "mu" else "pd", v)
            for L in Ls:
                df = None
                for d in dirs:
                    if (d / f"L{L}" / "runs").is_dir():
                        df = load_simple(d, seeds, T_SENS, [L])
                        if len(df) and df["complete"].any():
                            src = d
                            break
                        df = None
                if df is None or not len(df):
                    rows.append(dict(param=kind, value=v, L=L, n=0, n_expected=len(seeds)))
                    continue
                df = df[df["complete"]]
                k, n = int(df["runaway"].sum()), len(df)
                lo, hi = wilson(k, n)
                rows.append(dict(param=kind, value=v, L=L, n=n, n_expected=len(seeds), runaway_k=k,
                                 runaway_frac=k / n if n else np.nan, runaway_lo=lo, runaway_hi=hi,
                                 median_final_mean=float(df["final_mean"].median()) if n else np.nan,
                                 source=rel(src)))
    return pd.DataFrame(rows)


def snapshot(root: Path, L: int, seed: int, T: int):
    p = root / f"t{T}" / "original_mix" / f"L{L}" / "configs" / f"seed_{seed}_final.npy"
    return (np.load(p), p) if p.exists() else (None, p)


def largest_component(cfg: np.ndarray):
    occ = cfg > 0
    if not occ.any():
        return None, 0
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    sizes = ndimage.sum(occ, lab, range(1, n + 1))
    i = int(np.argmax(sizes)) + 1
    return lab == i, int(sizes[i - 1])


C_DEBRIS = 0.80     # grey of the cells outside a runaway giant


def render(cfg: np.ndarray, giant: np.ndarray | None = None) -> np.ndarray:
    """Cells coloured by type (golden-ratio hues). With `giant` (the mask of a runaway largest
    component) all other occupied cells are drawn grey so the giant's extent shows. The giant of
    the corrected model is a sparse network (almost every cell touches empty space), so neither
    painting nor outlining its edge cells shows anything but noise."""
    L = cfg.shape[0]
    img = np.ones((L, L, 3), np.float32)
    occ = cfg > 0
    if occ.any():
        types = cfg[occ].astype(np.float64)
        hues = (types * GOLDEN) % 1.0
        img[occ] = hsv_to_rgb(np.stack([hues, np.full_like(hues, 0.62), np.full_like(hues, 0.92)], -1))
        if giant is not None:
            img[occ & ~giant] = C_DEBRIS
    return img


def outline_segments(mask: np.ndarray) -> np.ndarray:
    """Cell-edge outline of a mask as (n, 2, 2) segments in imshow pixel coordinates (x=col, y=row)."""
    m = np.pad(mask, 1)
    r, c = np.nonzero(m[:, 1:] != m[:, :-1])       # vertical edges, between columns c-1 and c
    x, y = c - 0.5, r - 1.0
    v = np.stack([np.stack([x, y - 0.5], -1), np.stack([x, y + 0.5], -1)], 1)
    r, c = np.nonzero(m[1:, :] != m[:-1, :])       # horizontal edges, between rows r-1 and r
    x, y = c - 1.0, r - 0.5
    h = np.stack([np.stack([x - 0.5, y], -1), np.stack([x + 0.5, y], -1)], 1)
    return np.concatenate([v, h])


def draw_outline(ax, segs: np.ndarray, x0: int = 0, y0: int = 0, lw: float = 0.6):
    if len(segs):
        lc = LineCollection(segs - np.array([x0, y0]), colors="k", linewidths=lw, capstyle="projecting",
                            zorder=3)
        ax.add_collection(lc)


# ======================= drawing helpers =======================
def plabel(ax, s, dx=-0.20, dy=1.03):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            va="bottom", ha="right", color="#1a1a1a")


def snap_axis(ax, title, accent=None):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.1 if accent else 0.6)
        sp.set_color(accent if accent else "#b8b8b8")
    ax.set_title(title, fontsize=7, pad=2.5, color="#1a1a1a")


def pending(ax, what: str, where: str | None = None, frame=True):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_facecolor("#f5f5f5")
    for sp in ax.spines.values():
        sp.set_visible(frame)
        sp.set_color("#c4c4c4"); sp.set_linestyle((0, (3, 2))); sp.set_linewidth(0.7)
    txt = f"pending\n{what}" + (f"\n{where}" if where else "")
    ax.text(0.5, 0.5, txt, ha="center", va="center", transform=ax.transAxes,
            fontsize=5.8, color="#7a7a7a", linespacing=1.3, wrap=True)


def shade(ax, win):
    if win and np.isfinite(win.get("lo", np.nan)):
        ax.axvspan(win["lo"], win["hi"], color=BAND, zorder=0, lw=0)


def fmt_size(v: float) -> str:
    if not np.isfinite(v):
        return ""
    if v >= 1000:
        return f"{v / 1000:.1f}k" if v < 10000 else f"{v / 1000:.0f}k"
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"


def add_footnote(fig, notes: list[str]):
    if notes:
        fig.text(0.01, 0.004, "Preview, data incomplete: " + " ".join(notes), fontsize=5.4,
                 color="#9b2c2c", ha="left", va="bottom", wrap=True)


def partial_note(tab: pd.DataFrame, what: str) -> str | None:
    """Footnote for sizes that still miss runs (no file in either scan yet)."""
    p = tab[tab["n_used"] < tab["n_expected"]]
    if not len(p):
        return None
    return (f"{what}: L=" + ", ".join(f"{int(r.L)} ({int(r.n_used)}/{int(r.n_expected)} runs)" for r in p.itertuples())
            + " incomplete; the unfinished runs are mostly runaway (slower), so these sizes are biased low.")


KFMT = ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0")


def jitter(L, n, width):
    return L + np.random.default_rng(int(L)).uniform(-width, width, size=n)


# ======================= FIGURE: phenomenon =======================
def figure_phenomenon(a, runs, tab, legacy_tab, win, novelty, out_dir: Path) -> dict:
    status, notes = {}, []
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.97))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 1.22],
                           hspace=0.52, wspace=0.46, left=0.075, right=0.955, top=0.95, bottom=0.095)

    # (a,b) snapshots ------------------------------------------------------------
    row_specs = [(200, a.seed_a, "(a)", "compact", [("full", 5000), ("full", 10000), ("zoom", 10000)]),
                 (400, a.seed_b, "(b)", "runaway", [("full", 2000), ("full", 5000), ("full", 10000)])]
    snap_info = {}
    for row, (L, seed, lab, tag, spec) in enumerate(row_specs):
        miss = 0
        for col, (kind, T) in enumerate(spec):
            ax = fig.add_subplot(gs[row, col])
            cfg, path = snapshot(a.snapshot_root, L, seed, T)
            if cfg is None:
                miss += 1
                pending(ax, f"$L={L}$, seed {seed}, $t={tex_int(T)}$", short(path.parents[3]))
            else:
                big, smax = largest_component(cfg)
                fill = smax / L ** 2
                giant = big is not None and fill >= a.outline_max_fill
                snap_info[(L, T)] = dict(max_component=smax, n_occupied=int((cfg > 0).sum()), fill=fill,
                                         style="giant in colour, rest grey" if giant else "largest outlined")
                img = render(cfg, big if giant else None)
                segs = outline_segments(big) if (big is not None and not giant) else np.empty((0, 2, 2))
                if kind == "zoom":
                    z = a.zoom
                    if big is not None:
                        cy, cx = ndimage.center_of_mass(big)
                    else:
                        cy = cx = L / 2
                    y0 = int(np.clip(round(cy - z / 2), 0, L - z)); x0 = int(np.clip(round(cx - z / 2), 0, L - z))
                    snap_info[(L, T)]["zoom_origin"] = (y0, x0)
                    ax.imshow(img[y0:y0 + z, x0:x0 + z], interpolation="nearest", origin="upper")
                    draw_outline(ax, segs, x0, y0, lw=0.9)
                    ax.set_xlim(-0.5, z - 0.5); ax.set_ylim(z - 0.5, -0.5)
                    snap_axis(ax, f"$t={tex_int(T)}$  (zoom {z}$\\times${z})", accent=C_MAIN)
                    # mark the zoom window on the full t=T panel of the same row
                    for other in fig.axes:
                        if getattr(other, "_snap_key", None) == (L, T):
                            other.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), z, z, fill=False,
                                                      ec=C_MAIN, lw=0.9))
                else:
                    ax.imshow(img, interpolation="nearest", origin="upper")
                    draw_outline(ax, segs, lw=0.45)
                    ax.set_xlim(-0.5, L - 0.5); ax.set_ylim(L - 0.5, -0.5)
                    ax._snap_key = (L, T)
                    snap_axis(ax, f"$t={tex_int(T)}$   max$\\,{tex_int(smax)}$")
            if col == 0:
                ax.set_ylabel(f"$L={L}$\n({tag})", fontsize=8.5, fontweight="bold")
                ax.yaxis.label.set_visible(True)
                plabel(ax, lab, dx=-0.14, dy=1.0)
        status[lab] = "pending" if miss == len(spec) else ("partial" if miss else "rendered")
    if any(status[k] != "rendered" for k in ("(a)", "(b)")):
        notes.append("(a,b) snapshot configurations pending.")

    # (c) final mean size per run vs L ----------------------------------------------
    axc = fig.add_subplot(gs[2, 0])
    shade(axc, win)
    floor = 0.55
    use = runs[runs["complete"] | (runs["stopped"] & a.provisional)]
    present = set()
    for L, g in use.groupby("L"):
        grp = group_of(L, a)
        present.add(grp)
        col, mk = GROUPS[grp][:2]
        x = jitter(L, len(g), 2.6)
        comp = (g["complete"] & ~g["extinct"]).to_numpy()
        ext = g["extinct"].to_numpy()
        pend = g["stopped"].to_numpy()
        y = g["final_mean"].to_numpy()
        axc.scatter(x[comp], y[comp], s=6, color=col, marker=mk, alpha=0.45,
                    edgecolors="white", linewidths=0.2, zorder=3)
        if pend.any():
            axc.scatter(x[pend], y[pend], s=7, facecolors="none", edgecolors=col, marker=mk,
                        linewidths=0.5, alpha=0.7, zorder=3)
        if ext.any():
            axc.scatter(x[ext], np.full(ext.sum(), floor), s=7, color="#7a7a7a", marker="x",
                        linewidths=0.6, zorder=3)
    if len(tab):
        axc.plot(tab["L"], np.clip(tab["median_final_mean"], floor, None), color=C_MED, lw=1.5, zorder=5)
    axc.set_yscale("log")
    axc.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
    axc.yaxis.set_minor_locator(ticker.NullLocator())
    ymax = max(2e5, float(use["final_mean"].max()) * 40) if len(use) else 2e5
    axc.set_ylim(0.4, ymax)        # headroom above the runaway cluster for the legend
    axc.set_xlabel("Space size $L$"); axc.set_ylabel("Final mean component size")
    h = [Line2D([], [], color=C_MED, lw=1.5, label="Median")]
    h += [Line2D([], [], marker=GROUPS[k][1], color=GROUPS[k][0], lw=0, ms=GROUPS[k][2] - 0.1,
                 label=GROUPS[k][3]) for k in ("grid", "onset", "ref") if k in present]
    if (use["extinct"]).any():
        h.append(Line2D([], [], marker="x", color="#7a7a7a", lw=0, ms=3.5, mew=0.7, label="Extinct (0)"))
    if a.provisional and use["stopped"].any():
        h.append(Line2D([], [], marker="o", mfc="none", mec=C_MAIN, lw=0, ms=3.5, label="At early stop*"))
        notes.append(f"(c) and mechanism (a): {int(use['stopped'].sum())} runaway runs not yet re-run "
                     "without stop are shown at their early-stop value (open symbols).")
    axc.legend(handles=h, loc="upper left", frameon=False, handletextpad=0.35, borderpad=0.1,
               labelspacing=0.2, columnspacing=0.6, fontsize=5.8, ncol=2)
    sns.despine(ax=axc); plabel(axc, "(c)")
    status["(c)"] = "rendered" if len(use) else "pending"

    # (d) runaway fraction vs L ------------------------------------------------------
    axd = fig.add_subplot(gs[2, 1])
    shade(axd, win)
    if legacy_tab is not None and len(legacy_tab) and a.legacy:
        lt = legacy_tab.sort_values("L")
        axd.fill_between(lt["L"], lt["runaway_lo"], lt["runaway_hi"], color=C_LEG, alpha=0.18, lw=0)
        axd.plot(lt["L"], lt["runaway_frac"], "s-", color=C_LEG, ms=2.6, lw=1.0, mec="white", mew=0.3,
                 label="Earlier variant", zorder=2)
    t = tab[tab["n_used"] > 0].sort_values("L")
    if len(t):
        axd.fill_between(t["L"], t["runaway_lo"], t["runaway_hi"], color=C_MAIN, alpha=0.16, lw=0)
        axd.plot(t["L"], t["runaway_frac"], "-", color=C_MAIN, lw=1.3, zorder=3, label="Corrected model")
        for L, f, part in zip(t["L"], t["runaway_frac"], t["partial"]):
            col, mk, ms, _ = GROUPS[group_of(L, a)]
            if part:      # runs still missing: open marker (footnote)
                axd.plot(L, f, mk, mfc="white", mec=col, ms=ms, mew=0.9, zorder=4)
            else:
                axd.plot(L, f, mk, color=col, ms=ms, mec="white", mew=0.4, zorder=4)
        if np.isfinite(win.get("L50", np.nan)):
            Lg = np.linspace(t["L"].min(), t["L"].max(), 300)
            axd.plot(Lg, _logistic(Lg, win["pmax"], win["L50"], win["w"]), color=C_MED, lw=0.8,
                     ls=(0, (3, 2)), zorder=3, label="Logistic fit")
    axd.set_ylim(-0.03, 1.0)
    axd.set_xlabel("Space size $L$"); axd.set_ylabel("Runaway fraction")
    axd.legend(loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.1, fontsize=6.2,
               labelspacing=0.3)
    sns.despine(ax=axd); plabel(axd, "(d)")
    status["(d)"] = "rendered" if len(t) else "pending"
    pn = partial_note(tab, "(c)-(e) and open markers in (d)")
    if pn:
        notes.append(pn)

    # (e) long-run novelty at L=200 (two stacked panels) -------------------------------------
    sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 2], hspace=0.12)
    ax_p = fig.add_subplot(sub[0]); ax_s = fig.add_subplot(sub[1], sharex=ax_p)
    if novelty is None:
        pending(ax_p, "long-run novelty, $L=200$", short(a.novelty_root))
        pending(ax_s, "(metrics every step, 40 runs)")
        status["(e)"] = "pending"
        notes.append("(e) novelty runs pending.")
    else:
        st = novelty["step"].to_numpy()
        ax_p.fill_between(st, novelty["pat_q25"] / 1e3, novelty["pat_q75"] / 1e3, color=C_PAT, alpha=0.18, lw=0)
        ax_p.plot(st, novelty["pat_median"] / 1e3, color=C_PAT, lw=1.5)
        # the quantities are named inside the panels: two-line axis labels do not fit the panel height
        ax_p.text(0.03, 0.95, "Observed component-pattern\ntypes ($\\times10^3$)", transform=ax_p.transAxes,
                  fontsize=6.5, va="top")
        ax_p.set_ylim(0, float(novelty["pat_q75"].max()) / 1e3 * 1.45)
        plt.setp(ax_p.get_xticklabels(), visible=False)
        ax_s.fill_between(st, novelty["size_q25"], novelty["size_q75"], color=C_MAIN, alpha=0.18, lw=0)
        ax_s.plot(st, novelty["size_median"], color=C_MAIN, lw=1.5)
        ax_s.text(0.03, 0.95, "Mean component size", transform=ax_s.transAxes, fontsize=6.5, va="top")
        ax_s.set_ylim(0, max(7.0, float(novelty["size_q75"].max()) * 1.5))
        ax_s.set_xlabel("Time (steps)")
        ax_s.xaxis.set_major_formatter(KFMT)
        ax_p.set_title(f"$L=200$, median and IQR of {int(novelty['n'].iloc[0])} runs", fontsize=7)
        sns.despine(ax=ax_p); sns.despine(ax=ax_s)
        status["(e)"] = "rendered"
    plabel(ax_p, "(e)", dx=-0.24)

    add_footnote(fig, notes)
    out = out_dir / "fig_schc_phenomenon"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    plt.close(fig)
    print(f"Saved {rel(out)}.pdf / .png")
    return dict(status=status, notes=notes, snapshots=snap_info)


# ======================= FIGURE: mechanism =======================
CMAP_SENS = sns.color_palette("rocket_r", as_cmap=True)


def draw_phase(ax, sens: pd.DataFrame, kind: str, vals, Ls, title, ylabel):
    S = sens[sens["param"] == kind]
    F = np.full((len(vals), len(Ls)), np.nan)
    A = np.empty((len(vals), len(Ls)), dtype=object)
    for i, v in enumerate(vals):
        for j, L in enumerate(Ls):
            r = S[np.isclose(S["value"], v) & (S["L"] == L)]
            A[i, j] = ""
            if len(r) and r["n"].iloc[0] > 0:
                F[i, j] = r["runaway_frac"].iloc[0]
                A[i, j] = fmt_size(r["median_final_mean"].iloc[0]) + ("*" if r["n"].iloc[0] < r["n_expected"].iloc[0] else "")
    Fr, Ar = F[::-1], A[::-1]
    yl = [f"{v:g}" for v in vals][::-1]
    sns.heatmap(Fr, ax=ax, cmap=CMAP_SENS, vmin=0, vmax=1, annot=Ar, fmt="", annot_kws={"size": 6},
                linewidths=1.0, linecolor="white", xticklabels=Ls, yticklabels=yl, cbar=False)
    # missing cells: light hatch
    for i in range(Fr.shape[0]):
        for j in range(Fr.shape[1]):
            if not np.isfinite(Fr[i, j]):
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor="#f2f2f2", edgecolor="white", lw=1.0,
                                       hatch="////", zorder=0.5))
    ax.set_xlabel("Space size $L$"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7.8)
    ax.tick_params(left=False, bottom=False)
    plt.setp(ax.get_yticklabels(), rotation=0)
    return np.isfinite(F).sum()


def figure_mechanism(a, runs, tab, corr, corr_src, win, sens, out_dir: Path) -> dict:
    status, notes = {}, []
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 1.04))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.05, 0.92],
                           hspace=0.62, wspace=0.36, left=0.115, right=0.9, top=0.95, bottom=0.095)

    # (a) fill fraction of runaway runs at t = 20,000 ------------------------------------
    # Complete runs (t=20,000) and runs known only up to their early stop are kept apart:
    # the fill keeps growing after the stop (about 0.64 at the stop, 0.67 at t=20,000).
    axa = fig.add_subplot(gs[0, 0])
    comp = runs[runs["complete"] & runs["runaway"]]
    pend = runs[runs["stopped"]] if a.provisional else runs.iloc[0:0]
    allr = pd.concat([comp, pend])
    if len(allr):
        for L, g in allr.groupby("L"):
            col, mk = GROUPS[group_of(L, a)][:2]
            c = g[g["complete"]]; p = g[g["stopped"]]
            if len(c):
                axa.scatter(jitter(L, len(c), 3.0), c["fill"], s=10, color=col, marker=mk, alpha=0.6,
                            edgecolors="white", linewidths=0.3, zorder=3)
            if len(p):
                axa.scatter(jitter(L + 1, len(p), 3.0), p["fill"], s=9, facecolors="none", edgecolors=col,
                            marker=mk, linewidths=0.45, alpha=0.45, zorder=2)
        h = []
        mc = comp.groupby("L")["fill"].agg(["median", "size"])
        mc = mc[mc["size"] >= 3]
        if len(mc):
            axa.plot(mc.index, mc["median"], "-", color=C_MED, lw=1.3, zorder=4)
            h.append(Line2D([], [], color=C_MED, lw=1.3, label="Median, $t=20{,}000$"))
        if len(pend):
            mp = pend.groupby("L")["fill"].median()
            axa.plot(mp.index, mp.values, ls=(0, (2.5, 1.5)), color="#8a8a8a", lw=1.0, zorder=4)
            h.append(Line2D([], [], color="#8a8a8a", lw=1.0, ls=(0, (2.5, 1.5)), label="Median at early stop*"))
        if len(comp):
            h.append(Line2D([], [], marker="o", color=C_MAIN, lw=0, ms=3.5, label="Run at $t=20{,}000$"))
        if len(pend):
            h.append(Line2D([], [], marker="o", mfc="none", mec=C_MAIN, lw=0, ms=3.5, label="Run at early stop*"))
        if len(comp) >= 5:
            pooled = float(comp["fill"].median())
            axa.axhline(pooled, color="#9a9a9a", lw=0.8, ls=":", zorder=1)
            axa.text(0.03, pooled + 0.006, f"pooled median {pooled:.2f}", fontsize=6.2, color="#7a7a7a",
                     transform=axa.get_yaxis_transform(), va="bottom")
        lo = min(0.5, float(allr["fill"].min()) - 0.03)
        axa.set_ylim(lo, max(0.72, float(allr["fill"].max()) + 0.03))
        axa.legend(handles=h, loc="lower left", frameon=False, fontsize=6.0, handletextpad=0.3,
                   labelspacing=0.25, borderpad=0.1)
        status["(a)"] = "rendered" + (" (provisional)" if len(pend) else "")
    else:
        pending(axa, "runaway runs at $t=20{,}000$")
        status["(a)"] = "pending"
    xl = (runs["L"].min() - 10, runs["L"].max() + 10) if len(runs) else (190, 410)
    axa.set_xlim(*xl)
    axa.set_xlabel("Space size $L$")
    axa.set_ylabel(r"Fill fraction $s_{\max}/L^{2}$")
    axa.set_title("Runaway runs", fontsize=7.5)
    sns.despine(ax=axa); plabel(axa, "(a)")

    # (b) size-score correlation ---------------------------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    if corr is not None and len(corr):
        shade(axb, win)
        axb.axhline(0, color="#9a9a9a", lw=0.7, ls=":")
        c = corr.sort_values("L")
        prov = (c["r_status"] != "complete").to_numpy()
        # intervals drawn as [lo, hi] themselves: a percentile interval of r near -1 can exclude r
        axb.errorbar(c["L"], (c["r_lo"] + c["r_hi"]) / 2, yerr=(c["r_hi"] - c["r_lo"]) / 2, fmt="none",
                     ecolor=C_MAIN, alpha=0.5, elinewidth=1.0, capsize=1.8, zorder=2)
        axb.plot(c["L"], c["r"], "-", color=C_MAIN, lw=1.3, zorder=3)
        axb.plot(c["L"][~prov], c["r"][~prov], "o", color=C_MAIN, ms=3.6, mec="white", mew=0.4, zorder=4)
        axb.plot(c["L"][prov], c["r"][prov], "o", mfc="white", mec=C_MAIN, ms=3.6, mew=0.9, zorder=4)
        h = [Line2D([], [], color=C_MAIN, lw=1.3, marker="o", ms=3.6, mec="white", mew=0.4,
                    label=r"Ensemble $r$")]
        if a.r_per_run and c["r_run"].notna().any():
            axb.errorbar(c["L"] + 2, (c["r_run_lo"] + c["r_run_hi"]) / 2, yerr=(c["r_run_hi"] - c["r_run_lo"]) / 2,
                         fmt="none", ecolor=C_ADD, elinewidth=0.9, capsize=1.5, alpha=0.9, zorder=3)
            axb.plot(c["L"][~prov] + 2, c["r_run"][~prov], "D", color=C_ADD, ms=3.0, mec="white", mew=0.4,
                     zorder=4, ls="none")
            axb.plot(c["L"][prov] + 2, c["r_run"][prov], "D", mfc="white", mec=C_ADD, ms=3.0, mew=0.8,
                     zorder=4, ls="none")
            h.append(Line2D([], [], color=C_ADD, marker="D", ms=3.0, mec="white", mew=0.4, lw=0,
                            label=r"Per-run $r$ (mean Fisher $z$)"))
        if prov.any():
            h.append(Line2D([], [], color=C_MAIN, marker="o", mfc="white", ms=3.6, mew=0.9, lw=0,
                            label="Provisional*"))
            notes.append("(b) L=" + ", ".join(f"{int(r.L)} ({int(r.n_r)}/{len(a.seeds)})"
                                              for r in c[prov].itertuples())
                         + " provisional (open markers): only the runs with a complete trajectory, runaway "
                           "runs under-represented.")
        axb.set_ylim(-1.05, 1.0)
        axb.legend(handles=h, loc="upper right", frameon=False, fontsize=6.2, handletextpad=0.3)
        axb.set_xlim(*xl)
        status["(b)"] = f"rendered ({corr_src})"
    else:
        pending(axb, "size-score correlation")
        status["(b)"] = "pending"
    axb.set_xlabel("Space size $L$"); axb.set_ylabel("$r(\\bar s,\\bar q)$, steps 2,000\u201310,000")
    sns.despine(ax=axb); plabel(axb, "(b)")

    # (c,d) phase diagrams ---------------------------------------------------------------
    Ls = a.sens_L
    cax = fig.add_axes([0.915, 0.405, 0.016, 0.21])
    axc = fig.add_subplot(gs[1, 0]); axd = fig.add_subplot(gs[1, 1])
    nc = nd = 0
    has = lambda k: sens is not None and len(sens) and bool(((sens["param"] == k) & (sens["n"] > 0)).any())
    if has("mu"):
        nc = draw_phase(axc, sens, "mu", SENS_MU, Ls, r"Runaway vs $(L,\mu)$", r"Mutation probability $\mu$")
    else:
        pending(axc, r"$(L,\mu)$ sensitivity runs", short(a.sens_root))
    if has("p_death"):
        nd = draw_phase(axd, sens, "p_death", SENS_PD, Ls, r"Runaway vs $(L,p_{\mathrm{death}})$",
                        r"Death probability $p_{\mathrm{death}}$")
    else:
        pending(axd, r"$(L,p_{\mathrm{death}})$ sensitivity runs", short(a.sens_root))
    if nc + nd:      # one colour bar for both diagrams, whichever of them has data
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP_SENS), cax=cax)
        cb.set_label("Runaway fraction")
        cb.outline.set_visible(False)
        cb.ax.tick_params(size=2)
    else:
        cax.remove()
    if nc and sens is not None:
        fb = sens[(sens["param"] == "mu") & np.isclose(sens["value"], 0.002) & (sens["n"] > 0)
                  & sens.get("source", pd.Series("", index=sens.index)).astype(str).str.endswith("_mu0.002")]
        if len(fb):
            notes.append(r"(c) $\mu$=0.002 row at L=" + ", ".join(str(int(v)) for v in fb["L"])
                         + ": exact-mu runs (original_mix_mu0.002) until the baseline runs (mu=0.002/0.999) exist.")
    tot = len(SENS_MU) * len(Ls)
    status["(c)"] = "pending" if nc == 0 else ("rendered" if nc == tot else f"partial ({nc}/{tot} cells)")
    tot = len(SENS_PD) * len(Ls)
    status["(d)"] = "pending" if nd == 0 else ("rendered" if nd == tot else f"partial ({nd}/{tot} cells)")
    if nc < len(SENS_MU) * len(Ls) or nd < len(SENS_PD) * len(Ls):
        notes.append("(c,d) sensitivity runs pending" + (" (hatched cells; * = fewer runs than planned)."
                                                          if nc + nd else "."))
    plabel(axc, "(c)", dx=-0.17); plabel(axd, "(d)", dx=-0.17)

    # (e) nucleation kinetics --------------------------------------------------------------
    # the narrow right column stays empty: room for the line-end labels
    axe = fig.add_subplot(gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], width_ratios=[1, 0.1],
                                                           wspace=0.0)[0])
    Lnuc = [L for L in a.nuc_L if (runs["L"] == L).any()]
    # rocket over a wide range (larger L darker) plus alternating dash patterns, labelled at the
    # line ends: the six flare levels used before were too close to tell apart
    cols = sns.color_palette("rocket", as_cmap=True)(np.linspace(0.82, 0.12, max(len(Lnuc), 1)))
    styles = ["-", (0, (4.5, 1.6)), (0, (1.2, 1.2))]
    grid = np.arange(0, T_END + 1, 10)
    nuc_rows, ends = {}, []
    for i, L in enumerate(Lnuc):
        g = runs[(runs["L"] == L) & (runs["complete"] | runs["stopped"])]
        fc = g["first_cross"].to_numpy()
        frac = (fc[None, :] <= grid[:, None]).mean(1)
        nuc_rows[L] = frac
        lab = f"$L={L}$" + (f" ($n={len(g)}$)" if len(g) < len(a.seeds) else "")
        axe.plot(grid, frac, lw=1.6, color=cols[i], ls=styles[i % len(styles)], solid_capstyle="round",
                 dash_capstyle="round", zorder=3 + i * 0.01)
        ends.append([float(frac[-1]), lab, cols[i], styles[i % len(styles)]])
    if Lnuc:
        top = max(0.5, max(v.max() for v in nuc_rows.values()) * 1.15)
        axe.set_ylim(-0.01, top)
        # direct labels at the right end, pushed apart to a minimum vertical gap (data units)
        gap = 0.075 * top
        order = sorted(range(len(ends)), key=lambda k: ends[k][0])
        ys = [ends[k][0] for k in order]
        for j in range(1, len(ys)):
            ys[j] = max(ys[j], ys[j - 1] + gap)
        shift = max(0.0, ys[-1] - (top - 0.02 * top)) if ys else 0.0
        ys = [y - shift for y in ys]
        for j in range(len(ys) - 2, -1, -1):          # keep the gap after pulling the top down
            ys[j] = min(ys[j], ys[j + 1] - gap)
        for k, y in zip(order, ys):
            y0, lab, col, ls = ends[k]
            axe.plot([T_END, T_END + 380], [y0, y], color=col, lw=0.6, clip_on=False)
            axe.plot([T_END + 380, T_END + 1150], [y, y], color=col, lw=1.6, ls=ls, clip_on=False,
                     solid_capstyle="butt")
            axe.text(T_END + 1300, y, lab, color=col, fontsize=6.6, va="center", ha="left", clip_on=False)
        axe.text(0.012, 0.97, "Nucleation kinetics (nucleated: mean component size > 100)", transform=axe.transAxes,
                 fontsize=7.5, va="top")
        status["(e)"] = "rendered (L=" + ",".join(map(str, Lnuc)) + ")"
    else:
        pending(axe, "nucleation kinetics")
        status["(e)"] = "pending"
    missing = [L for L in a.nuc_L if L not in Lnuc]
    if missing:
        notes.append("(e) L=" + ", ".join(map(str, missing)) + " not yet available.")
    axe.set_xlim(0, T_END)
    axe.xaxis.set_major_formatter(KFMT)
    axe.set_xlabel("Time (steps)"); axe.set_ylabel("Fraction nucleated")
    sns.despine(ax=axe); plabel(axe, "(e)", dx=-0.075)

    pn = partial_note(tab[tab["L"].isin(set(Lnuc) | set(runs.loc[runs["runaway"], "L"]))], "(a,e)")
    if pn:
        notes.append(pn)
    add_footnote(fig, notes)
    out = out_dir / "fig_schc_mechanism"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    plt.close(fig)
    print(f"Saved {rel(out)}.pdf / .png")
    nuc = pd.DataFrame({"step": grid[::10], **{f"L{L}": v[::10] for L, v in nuc_rows.items()}})
    return dict(status=status, notes=notes, nucleation=nuc)


# ======================= report =======================
def md_table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if c in fmt and pd.notna(v) and v != "":
                cells.append(fmt[c].format(v))
            elif isinstance(v, (float, np.floating)) and np.isnan(v):
                cells.append("")
            elif isinstance(v, (float, np.floating)) and float(v).is_integer() and c not in fmt:
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(a, out_dir, runs, tab, legacy_tab, win, corr, corr_src, sens, novelty, ph, me):
    L_ = []
    L_.append("# Extension II figures (corrected model): data status and key numbers\n")
    L_.append(f"Generated by `scripts/npj_rev_ext2_figs.py` on {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
              f"Provisional merge of early-stopped runs: {'on' if a.provisional else 'off'}.\n")
    L_.append("## Panel status\n")
    for name, st in (("fig_schc_phenomenon", ph["status"]), ("fig_schc_mechanism", me["status"])):
        L_.append(f"- **{name}**: " + "; ".join(f"{k} {v}" for k, v in st.items()))
    notes = ph["notes"] + me["notes"]
    if notes:
        L_.append("\nFigure footnotes (shown on the figures while data are incomplete):\n")
        L_ += [f"- {n}" for n in notes]
    L_.append("\n## Size scan per L\n")
    L_.append(f"Sources: complete runs from `{rel(a.full_root)}`, plus runaway runs stopped at mean size "
              f"{STOP:.0f} in `{rel(a.stop_root)}` that still await the re-run (n_stopped_pending; counted as runaway "
              "when provisional). Seeds " + f"{a.seeds[0]}-{a.seeds[-1]}.\n")
    L_.append("Median in phenomenon (c): complete runs plus pending runs at their early-stop value (> 100, so the "
              "median is unaffected while the runaway fraction is below 0.5). Fill medians: complete runaway runs at "
              "t=20,000 (fill_median_runaway_t20k) and with the pending runs at their stop "
              "(fill_median_runaway_with_pending).\n")
    L_.append(md_table(tab, ["L", "n_complete", "n_stopped_pending", "n_missing", "runaway_k", "n_used",
                             "runaway_frac", "runaway_lo", "runaway_hi", "median_final_mean", "n_extinct",
                             "n_fell_back", "fill_median_runaway_t20k", "fill_median_runaway_with_pending"],
                       {"runaway_frac": "{:.2f}", "runaway_lo": "{:.2f}", "runaway_hi": "{:.2f}",
                        "median_final_mean": "{:.2f}", "fill_median_runaway_t20k": "{:.3f}",
                        "fill_median_runaway_with_pending": "{:.3f}"}))
    L_.append("\n## Transition window\n")
    if np.isfinite(win.get("L50", np.nan)):
        L_.append(f"Logistic fit p(L) = pmax / (1 + exp(-(L - L50)/w)) (binomial ML, all L): pmax = {win['pmax']:.3f}, "
                  f"L50 = {win['L50']:.1f} (95% parametric bootstrap {win['L50_lo']:.1f}-{win['L50_hi']:.1f}), "
                  f"w = {win['w']:.1f}. 10-90% rise: {win['L50'] - win['w'] * np.log(9):.1f}-"
                  f"{win['L50'] + win['w'] * np.log(9):.1f} (bootstrap of the edges: lower {win['win_lo_lo']:.0f}-"
                  f"{win['win_lo_hi']:.0f}, upper {win['win_hi_lo']:.0f}-{win['win_hi_hi']:.0f}).")
    L_.append(f"\nShaded window ({win['method']}): {win['lo']:.1f}-{win['hi']:.1f}." if np.isfinite(win["lo"])
              else "\nNo window shaded.")
    if legacy_tab is not None and len(legacy_tab):
        L_.append("\n## Published model re-run with the CPU engine (grey in phenomenon (d))\n")
        L_.append(md_table(legacy_tab, ["L", "n", "runaway_k", "runaway_frac", "runaway_lo", "runaway_hi"],
                           {"runaway_frac": "{:.2f}", "runaway_lo": "{:.2f}", "runaway_hi": "{:.2f}"}))
    if corr is not None and len(corr):
        L_.append("\n## Size-score correlation (mechanism (b))\n")
        L_.append(f"Sources: {corr_src}.\n")
        cols = ["L", "n_r", "r", "r_lo", "r_hi", "r_run", "r_run_lo", "r_run_hi", "r_run_median", "r_status"]
        L_.append(md_table(corr, cols, {c: "{:.3f}" for c in cols if c.startswith("r_") and c != "r_status"} | {"r": "{:.3f}"}))
        L_.append("\nDefinitions and bootstrap exactly as Table 2 (`scripts/npj_rev_ext2_tables.py`, `table2_row`): "
                  "r = Pearson r of the ensemble-mean mean_size trajectory (extinct runs contribute size 0) and the "
                  "ensemble-mean mean_fitness trajectory (averaged over the runs alive at each sample), steps "
                  "2,000-10,000; 95% percentile bootstrap over runs, 10,000 resamples, weights from "
                  "default_rng([20260925, L]). r_run = per-run r on the live samples, back-transformed mean Fisher z "
                  "with the same bootstrap; r_run_median = median per-run r. Rows of table2.csv with size_status "
                  "'complete' are read as they are; the other L are computed here with the same functions. "
                  "r_status 'provisional': fewer than all runs have a complete trajectory (runaway runs "
                  "under-represented), open markers in the figure.")
    L_.append("\n## Snapshots\n")
    if ph["snapshots"]:
        for (L, T), d in sorted(ph["snapshots"].items()):
            L_.append(f"- L={L}, t={T}: largest component {d['max_component']:,} cells (fill {d['fill']:.3f}), "
                      f"occupied {d['n_occupied']:,}; drawn as: {d['style']}"
                      + (f", zoom origin (row, col) {d['zoom_origin']}" if "zoom_origin" in d else ""))
        L_.append(f"\nRendering: cells coloured by type. A largest component below fill {a.outline_max_fill:g} is "
                  "outlined cell by cell in black; above it (the sparse runaway network, whose cells nearly all "
                  "touch empty space) it keeps its type colours and all other occupied cells are grey.")
    else:
        L_.append(f"- none yet (`{rel(a.snapshot_root)}`)")
    L_.append("\n## Novelty (phenomenon (e))\n")
    if novelty is not None:
        e = novelty.iloc[-1]
        L_.append(f"- {int(e['n'])} runs; at t={int(e['step'])}: median cumulative pattern types {e['pat_median']:.0f}, "
                  f"median mean size {e['size_median']:.2f}")
    else:
        L_.append(f"- pending (`{rel(a.novelty_root)}`)")
    L_.append("\n## Sensitivity (mechanism (c,d))\n")
    L_.append("The mu = 0.002 row and the p_death = 0.001 row are the same baseline run set `original_mix` "
              "(mu = 0.002/0.999, p_death = 0.001), so the shared cells agree between (c) and (d); "
              "`original_mix_mu0.002` (mu exactly 0.002) is used for the mu row only while the baseline is "
              "missing (see `source`).\n")
    if sens is not None and (sens["n"] > 0).any():
        L_.append(md_table(sens[sens["n"] > 0], ["param", "value", "L", "n", "runaway_k", "runaway_frac",
                                                 "median_final_mean", "source"],
                           {"runaway_frac": "{:.2f}", "median_final_mean": "{:.2f}"}))
    else:
        L_.append(f"- pending (`{rel(a.sens_root)}`)")
    fb = runs[runs["fell_back"]]
    L_.append("\n## Data checks\n")
    L_.append(f"- Runs that crossed mean size 100 and ended at or below it (complete runs): {len(fb)}"
              + (" (" + ", ".join(f"L={r.L} seed {r.seed}" for r in fb.itertuples()) + ")" if len(fb) else ""))
    inc = runs[runs["source"].str.contains("incomplete")]
    L_.append(f"- Runs neither complete nor stopped at the threshold (ignored): {len(inc)}")
    (out_dir / "report.md").write_text("\n".join(L_) + "\n")


# ======================= main =======================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    R = ROOT / "results"
    ap.add_argument("--out", type=Path, default=R / "npj_rev" / "ext2fig")
    ap.add_argument("--full-root", type=Path, default=R / "cpu_scan_full" / "original_mix")
    ap.add_argument("--stop-root", type=Path, default=R / "cpu_scan" / "original_mix")
    ap.add_argument("--legacy-root", type=Path, default=R / "cpu_scan" / "legacy_xor")
    ap.add_argument("--novelty-root", type=Path, default=R / "cpu_novelty" / "original_mix" / "L200")
    ap.add_argument("--snapshot-root", type=Path, default=R / "cpu_snapshots")
    ap.add_argument("--sens-root", type=Path, default=R / "cpu_sensitivity")
    ap.add_argument("--table2", type=Path, default=R / "npj_rev" / "ext2_tables" / "table2.csv",
                    help="Table 2 of scripts/npj_rev_ext2_tables.py (mechanism (b) values for complete L)")
    ap.add_argument("--r-source", choices=["auto", "compute"], default="auto",
                    help="auto: complete rows of --table2, the other L computed with the tables script's "
                         "functions; compute: every L computed here (same functions, same numbers)")
    ap.add_argument("--seeds", default="0-99", help="seeds of the size scan")
    ap.add_argument("--sens-seeds", default="0-29")
    ap.add_argument("--sens-L", type=int, nargs="+", default=[100, 200, 240, 300, 320, 400])
    ap.add_argument("--coarse-L", type=int, nargs="+", default=list(range(200, 401, 20)),
                    help="sizes drawn as the Delta L=20 grid (blue circles)")
    ap.add_argument("--onset-L", type=int, nargs="+", default=[120, 140, 160, 180, 210, 230, 250],
                    help="sizes drawn as the onset scan (orange diamonds)")
    ap.add_argument("--ref-L", type=int, nargs="+", default=[100],
                    help="original SCHC reference size(s) (blue triangle)")
    ap.add_argument("--outline-max-fill", type=float, default=0.05,
                    help="outline the largest component in the snapshots only below this fill s_max/L^2; "
                         "above it the giant keeps its colours and the other cells are grey")
    ap.add_argument("--nuc-L", type=int, nargs="+", default=[200, 220, 240, 260, 320, 400])
    ap.add_argument("--window", nargs="+", default=["auto"],
                    help="'auto' (10-90%% rise of a logistic fit to the runaway fraction), 'none', or LO HI")
    ap.add_argument("--zoom", type=int, default=60)
    ap.add_argument("--seed-a", type=int, default=0, help="seed of the L=200 snapshots")
    ap.add_argument("--seed-b", type=int, default=5, help="seed of the L=400 snapshots")
    ap.add_argument("--no-provisional", dest="provisional", action="store_false",
                    help="ignore runaway runs that still await their no-stop re-run")
    ap.add_argument("--no-legacy", dest="legacy", action="store_false")
    ap.add_argument("--no-r-per-run", dest="r_per_run", action="store_false")
    ap.add_argument("--only", choices=["phenomenon", "mechanism"], default=None)
    a = ap.parse_args()
    a.seeds = parse_seeds(a.seeds)
    a.out.mkdir(parents=True, exist_ok=True)
    if len(a.window) == 1 and a.window[0] in ("auto", "none"):
        win_spec = a.window[0]
    elif len(a.window) == 2:
        try:
            win_spec = (float(a.window[0]), float(a.window[1]))
        except ValueError:
            ap.error("--window LO HI takes two numbers")
    else:
        ap.error("--window takes 'auto', 'none' or two numbers LO HI")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("loading size scan ...", flush=True)
    runs = load_scan(a.full_root, a.stop_root, a.seeds)
    if not len(runs):
        raise SystemExit(f"no runs under {a.full_root} or {a.stop_root}")
    other = sorted(set(runs["L"]) - set(a.coarse_L) - set(a.onset_L) - set(a.ref_L))
    if other:
        print(f"warning: L={other} are in none of --coarse-L/--onset-L/--ref-L; drawn as the onset scan")
    tab = by_L_table(runs, a.seeds, a.provisional)
    win = transition_window(tab, win_spec)
    leg = None
    if a.legacy and a.legacy_root.is_dir():
        ld = load_simple(a.legacy_root, a.seeds, T_END)
        if len(ld):
            ld = ld[ld["usable"]]
            leg = ld.groupby("L").agg(n=("runaway", "size"), runaway_k=("runaway", "sum")).reset_index()
            leg["runaway_frac"] = leg["runaway_k"] / leg["n"]
            leg[["runaway_lo", "runaway_hi"]] = [wilson(int(k), int(n)) for k, n in zip(leg["runaway_k"], leg["n"])]
    print("correlations ...", flush=True)
    corr, corr_src = correlations(runs["L"].unique(), a)
    print("novelty / sensitivity ...", flush=True)
    novelty = load_novelty(a.novelty_root)
    sens = load_sensitivity(a.sens_root, a.sens_L, parse_seeds(a.sens_seeds))

    runs.sort_values(["L", "seed"]).to_csv(a.out / "runs.csv", index=False)
    tab.merge(corr, on="L", how="left").to_csv(a.out / "by_L.csv", index=False)
    pd.DataFrame([win]).to_csv(a.out / "window_fit.csv", index=False)
    sens.to_csv(a.out / "sensitivity.csv", index=False)
    if novelty is not None:
        novelty[novelty["step"] % 10 == 0].to_csv(a.out / "novelty.csv", index=False)   # every 10 steps
    if leg is not None:
        leg.to_csv(a.out / "legacy_by_L.csv", index=False)

    ph = dict(status={}, notes=[], snapshots={})
    me = dict(status={}, notes=[], nucleation=None)
    if a.only in (None, "phenomenon"):
        ph = figure_phenomenon(a, runs, tab, leg, win, novelty, a.out)
    if a.only in (None, "mechanism"):
        me = figure_mechanism(a, runs, tab, corr, corr_src, win, sens, a.out)
        me["nucleation"].to_csv(a.out / "nucleation.csv", index=False)
    write_report(a, a.out, runs, tab, leg, win, corr, corr_src, sens, novelty, ph, me)
    print(f"report: {rel(a.out / 'report.md')}")


if __name__ == "__main__":
    main()
