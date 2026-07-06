#!/usr/bin/env python3
"""Build the two SCHC case-study figures for the npj Complexity review.

All panels are redrawn from the raw per-seed CSVs and saved config arrays
(no image reuse). Two of the panels are new encodings not present in the
conference paper: the (L, mu)/(L, p_death) runaway phase diagrams and the
nucleation-kinetics curves.

Figure 2  (fig_schc_phenomenon): the scale-controlled transition
  (a) L=200 snapshots (compact)   (b) L=400 snapshots (runaway)
  (c) final mean size vs L        (d) runaway fraction vs L
  (e) long-run novelty: cumulative pattern types rise while size saturates

Figure 3  (fig_schc_mechanism): stochastic onset and mechanism
  (a) boundary control (open vs periodic)   (b) size-score correlation vs L
  (c) phase diagram (L x mu)                (d) phase diagram (L x p_death)
  (e) nucleation kinetics: fraction of runs nucleated vs time, by L
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "tex" / "Large-Hash-Chemistry" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "figure.dpi": 300,
})
MM = 1 / 25.4
DOUBLE_COL = 180 * MM
C_OPEN = "#0072B2"
C_PERI = "#D55E00"
C_MED = "#222222"
GOLDEN = 0.618033988749895
RUN = 100.0
SNAP_STEPS = (6000, 10000, 20000)

OPEN = ROOT / "results" / "transition_scan"
FINE = ROOT / "results" / "fine_transition_scan"
PERI = ROOT / "results" / "boundary_control" / "periodic_v2"
LARGE = ROOT / "results" / "large_space"
SNAP = ROOT / "results" / "snapshots_hero"


# ---------- helpers ----------
def render(cfg):
    L = cfg.shape[0]
    img = np.ones((L, L, 3), np.float32)
    occ = cfg > 0
    if occ.any():
        types = cfg[occ].astype(np.float64)
        hues = (types * GOLDEN) % 1.0
        img[occ] = hsv_to_rgb(np.stack([hues, np.full_like(hues, 0.8),
                                        np.full_like(hues, 0.85)], -1))
        lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
        if n:
            big = lab == int(np.argmax(ndimage.sum(occ, lab, range(1, n + 1)))) + 1
            img[big & ~ndimage.binary_erosion(big, structure=np.ones((3, 3)))] = 0.0
    return img


def maxcomp(cfg):
    occ = cfg > 0
    if not occ.any():
        return 0
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    return int(ndimage.sum(occ, lab, range(1, n + 1)).max()) if n else 0


def finals(root, L):
    d = root / f"L{L}" / "runs"
    if not d.is_dir():
        return None
    v = [float(pd.read_csv(f)["mean_size"].iloc[-1]) for f in sorted(d.glob("seed_*.csv"))]
    return np.array(v, float) if v else None


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def load_cfg(L, seed, step):
    p = ROOT / "results" / "gifs" / f"configs_L{L}_seed{seed}" / f"config_step{step:05d}.npy"
    return np.load(p) if p.exists() else None


def plabel(ax, s, dx=-0.16, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")


# ======================= FIGURE 2: phenomenon =======================
def figure_phenomenon():
    # Per-row snapshot specs: ("full", step) or ("zoom", step, (x0,x1,y0,y1)).
    # L=200 stays compact (t=2000 has only ~5 active cells, so we show the
    # populated t=5000/10000 plus a zoom that reveals the small structures);
    # L=400 shows the runaway progression clusters -> spreading -> fill.
    row_specs = [
        (200, 0, "(a)", "compact", [("full", 5000), ("full", 10000),
                                    ("zoom", 10000, (70, 130, 70, 130))]),
        (400, 8, "(b)", "runaway", [("full", 2000), ("full", 5000), ("full", 10000)]),
    ]

    coarse = {L: finals(OPEN, L) for L in range(200, 401, 20)}
    coarse = {L: v for L, v in coarse.items() if v is not None}
    fine = {L: finals(FINE, L) for L in range(300, 321, 2)}
    fine = {L: v for L, v in fine.items() if v is not None}

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.92))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 1.15],
                           hspace=0.5, wspace=0.42,
                           left=0.075, right=0.945, top=0.95, bottom=0.085)

    for row, (L, seed, lab, tag, spec) in enumerate(row_specs):
        for col, item in enumerate(spec):
            ax = fig.add_subplot(gs[row, col])
            cfg = load_cfg(L, seed, item[1])
            if cfg is not None:
                if item[0] == "zoom":
                    x0, x1, y0, y1 = item[2]
                    crop = cfg[x0:x1, y0:y1]
                    ax.imshow(render(crop), interpolation="nearest", origin="upper")
                    ax.set_title(f"$t={item[1]:,}$  (zoom {x1-x0}$\\times${y1-y0})",
                                 fontsize=7, pad=2)
                    for sp in ax.spines.values():
                        sp.set_linewidth(0.9); sp.set_color(C_OPEN)
                else:
                    ax.imshow(render(cfg), interpolation="nearest", origin="upper")
                    ax.set_title(f"$t={item[1]:,}$  (max {maxcomp(cfg):,})", fontsize=7, pad=2)
                    for sp in ax.spines.values():
                        sp.set_linewidth(0.6); sp.set_color("0.6")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"$L={L}$\n({tag})", fontsize=8, fontweight="bold")
                plabel(ax, lab, dx=-0.13, dy=1.02)

    # (c) final mean size vs L
    axc = fig.add_subplot(gs[2, 0])
    axc.axvspan(300, 320, color="0.88", zorder=0)
    for L, v in coarse.items():
        axc.scatter(np.full_like(v, L), np.clip(v, 1, None), s=6, color=C_OPEN,
                    alpha=0.35, edgecolors="none", zorder=2)
    axc.plot(sorted(coarse), [np.median(coarse[L]) for L in sorted(coarse)], "-",
             color=C_MED, lw=1.3, zorder=4, label="Median")
    for L, v in fine.items():
        axc.scatter(np.full_like(v, L), np.clip(v, 1, None), s=5, color=C_PERI,
                    alpha=0.5, marker="s", edgecolors="none", zorder=3)
    axc.scatter([], [], s=6, color=C_OPEN, label=r"Coarse $\Delta L{=}20$")
    axc.scatter([], [], s=5, color=C_PERI, marker="s", label=r"Fine $\Delta L{=}2$")
    axc.set_yscale("log"); axc.set_ylim(0.8, 2e5)
    axc.set_xlabel("Space size $L$"); axc.set_ylabel("Final mean size")
    axc.legend(loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.2)
    plabel(axc, "(c)")

    # (d) runaway fraction vs L
    axd = fig.add_subplot(gs[2, 1])
    axd.axvspan(300, 320, color="0.88", zorder=0)
    for src, color, mk, lab in [(coarse, C_OPEN, "o", "Coarse"), (fine, C_PERI, "s", "Fine")]:
        Ls = sorted(src); fr = []; lo = []; hi = []
        for L in Ls:
            v = src[L]; k = int((v > RUN).sum()); n = len(v)
            fr.append(k / n); a, b = wilson(k, n); lo.append(fr[-1] - a); hi.append(b - fr[-1])
        axd.errorbar(Ls, fr, yerr=[lo, hi], fmt=mk + "-", color=color, ms=3.5,
                     lw=1.0, capsize=1.5, elinewidth=0.7, label=lab)
    axd.set_ylim(-0.05, 1.05)
    axd.set_xlabel("Space size $L$"); axd.set_ylabel("Runaway fraction")
    axd.legend(loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.2)
    plabel(axd, "(d)")

    # (e) long-run novelty: patterns rise while size saturates (L=200)
    axe = fig.add_subplot(gs[2, 2])
    files = sorted((LARGE / "L200" / "runs").glob("seed_*.csv"))[:40]
    dfs = [pd.read_csv(f)[["step", "mean_size", "cum_pattern_types"]] for f in files]
    steps = dfs[0]["step"].values
    pat = np.median(np.stack([d["cum_pattern_types"].values for d in dfs]), 0)
    sz = np.median(np.stack([d["mean_size"].values for d in dfs]), 0)
    l1, = axe.plot(steps, pat, color=C_PERI, lw=1.4, label="Cum. pattern types")
    axe.set_xlabel("Time (steps)"); axe.set_ylabel("Cum. pattern types", color=C_PERI)
    axe.tick_params(axis="y", labelcolor=C_PERI)
    axe.set_ylim(0, None)
    ax2 = axe.twinx()
    l2, = ax2.plot(steps, sz, color=C_OPEN, lw=1.4, label="Mean size")
    ax2.set_ylabel("Mean size", color=C_OPEN); ax2.tick_params(axis="y", labelcolor=C_OPEN)
    ax2.set_ylim(0, max(6, sz.max() * 1.4))
    axe.set_title(r"$L=200$: novelty persists, size saturates", fontsize=7)
    axe.legend(handles=[l1, l2], loc="upper left", frameon=False, fontsize=6.5)
    plabel(axe, "(e)")

    out = FIG_DIR / "fig_schc_phenomenon"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png", dpi=300)
    print(f"Saved {out}.pdf / .png")


# ======================= FIGURE 3: mechanism =======================
def phase_grid(root, param_col, param_vals, Ls, summ_csv):
    df = pd.read_csv(summ_csv)
    G = np.full((len(param_vals), len(Ls)), np.nan)
    for i, pv in enumerate(param_vals):
        for j, L in enumerate(Ls):
            row = df[(np.isclose(df[param_col], pv)) & (df["L"] == L)]
            if len(row):
                G[i, j] = float(row["mean_size_final"].iloc[0])
    return G


def draw_phase(ax, G, param_vals, Ls, plabel_txt, ptitle):
    im = ax.imshow(np.log10(np.clip(G, 1, None)), origin="lower", aspect="auto",
                   cmap="magma", vmin=0, vmax=5)
    ax.set_xticks(range(len(Ls))); ax.set_xticklabels(Ls)
    ax.set_yticks(range(len(param_vals))); ax.set_yticklabels(param_vals)
    ax.set_xlabel("Space size $L$")
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            val = G[i, j]
            if not np.isnan(val):
                txt = f"{val:.0f}" if val < 1000 else f"{val/1000:.0f}k"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color="white" if np.log10(max(val, 1)) < 3.2 else "black")
    ax.set_title(ptitle, fontsize=7.5)
    return im


def figure_mechanism():
    bnd_L = [200, 240, 280, 300, 320, 360]
    bo = {L: finals(OPEN, L) for L in bnd_L}
    bp = {L: finals(PERI, L) for L in bnd_L}
    ta = pd.read_csv(OPEN / "analysis" / "transition_analysis.csv")

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 1.0))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 0.95],
                           hspace=0.6, wspace=0.32,
                           left=0.10, right=0.93, top=0.94, bottom=0.09)

    # (a) boundary control
    axa = fig.add_subplot(gs[0, 0])
    Ls = [L for L in bnd_L if bo.get(L) is not None]
    axa.plot(Ls, [np.median(bo[L]) for L in Ls], "o-", color=C_OPEN, ms=5, lw=1.4,
             label="Open")
    axa.plot(Ls, [np.median(bp[L]) if bp.get(L) is not None else np.nan for L in Ls],
             "s--", color=C_PERI, ms=5, lw=1.4, label="Periodic")
    axa.axhline(RUN, color="0.6", lw=0.7, ls=":")
    axa.set_yscale("log"); axa.set_ylim(1, 5e3)
    axa.set_xlabel("Space size $L$"); axa.set_ylabel("Median final mean size")
    axa.legend(loc="center right", frameon=False, title="Boundary")
    plabel(axa, "(a)")

    # (b) size-score correlation
    axb = fig.add_subplot(gs[0, 1])
    axb.axvspan(300, 320, color="0.88", zorder=0)
    axb.axhline(0, color="0.6", lw=0.6, ls=":")
    axb.plot(ta["L"], ta["corr_size_fitness_2k_10k"], "o-", color=C_OPEN, ms=3.5, lw=1.0)
    axb.set_ylim(-1.0, 0.5)
    axb.set_xlabel("Space size $L$"); axb.set_ylabel(r"Corr($\bar s,\bar q$)")
    plabel(axb, "(b)")

    # (c,d) phase diagrams
    axc = fig.add_subplot(gs[1, 0])
    mu_vals = [0.001, 0.002, 0.005, 0.01]; L4 = [200, 300, 320, 400]
    Gmu = phase_grid(None, "mu", mu_vals, L4,
                     ROOT / "results/param_sensitivity_mu/sensitivity_summary_mu.csv")
    draw_phase(axc, Gmu, mu_vals, L4, "(c)", r"Final mean size vs $(L,\mu)$")
    axc.set_ylabel(r"Mutation rate $\mu$")
    plabel(axc, "(c)")

    axd = fig.add_subplot(gs[1, 1])
    d_vals = [0.0005, 0.001, 0.005, 0.01]
    Gd = phase_grid(None, "death_prob", d_vals, L4,
                    ROOT / "results/param_sensitivity_death/sensitivity_summary_death_prob.csv")
    im = draw_phase(axd, Gd, d_vals, L4, "(d)", r"Final mean size vs $(L,p_{\mathrm{death}})$")
    axd.set_ylabel(r"Death prob. $p_{\mathrm{death}}$")
    plabel(axd, "(d)")
    cb = fig.colorbar(im, ax=[axc, axd], fraction=0.046, pad=0.02, location="right")
    cb.set_label(r"$\log_{10}$ final mean size", fontsize=7)

    # (e) nucleation kinetics
    axe = fig.add_subplot(gs[2, :])
    cmap = plt.cm.viridis
    L_nuc = [320, 340, 360, 400]
    for idx, L in enumerate(L_nuc):
        d = OPEN / f"L{L}" / "runs"
        cross = []
        n = 0
        for f in sorted(d.glob("seed_*.csv")):
            df = pd.read_csv(f); n += 1
            over = df[df["mean_size"] > RUN]
            cross.append(int(over["step"].iloc[0]) if len(over) else np.inf)
        grid = np.linspace(0, 20000, 200)
        frac = [np.mean([c <= t for c in cross]) for t in grid]
        axe.plot(grid, frac, lw=1.5, color=cmap(idx / (len(L_nuc) - 1)), label=f"$L={L}$")
    axe.set_xlabel("Time (steps)"); axe.set_ylabel("Fraction nucleated")
    axe.set_ylim(-0.03, 1.03)
    axe.legend(loc="upper left", frameon=False, ncol=2, title="Nucleation kinetics")
    plabel(axe, "(e)", dx=-0.06)

    out = FIG_DIR / "fig_schc_mechanism"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png", dpi=300)
    print(f"Saved {out}.pdf / .png")


if __name__ == "__main__":
    figure_phenomenon()
    figure_mechanism()
