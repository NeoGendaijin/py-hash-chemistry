#!/usr/bin/env python3
"""Build the two SCHC case-study figures for the npj Complexity review.

Seaborn-styled, publication-grade. All panels are redrawn from the raw
per-seed CSVs and saved config arrays (no image reuse). Two panels are new
encodings not present in the conference paper: the (L, mu)/(L, p_death)
runaway phase diagrams and the nucleation-kinetics curves.

DATA is never modified here -- only presentation/encoding.

Figure 2  fig_schc_phenomenon : the scale-controlled transition
Figure 3  fig_schc_mechanism  : stochastic onset and mechanism
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
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "tex" / "Large-Hash-Chemistry" / "figures" / "ext2-scale"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- seaborn theme ----------------
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
C_OPEN, C_PERI, C_MED = PAL[0], PAL[3], "#2b2b2b"   # blue, red, near-black
BAND = "#e9e6ef"                                     # transition-window shade
MM = 1 / 25.4
DOUBLE_COL = 180 * MM
GOLDEN = 0.618033988749895
RUN = 100.0

OPEN = ROOT / "results" / "transition_scan"
FINE = ROOT / "results" / "fine_transition_scan"
PERI = ROOT / "results" / "boundary_control" / "periodic_v2"
LARGE = ROOT / "results" / "large_space"


# ---------------- helpers ----------------
def render(cfg):
    L = cfg.shape[0]
    img = np.ones((L, L, 3), np.float32)
    occ = cfg > 0
    if occ.any():
        types = cfg[occ].astype(np.float64)
        hues = (types * GOLDEN) % 1.0
        img[occ] = hsv_to_rgb(np.stack([hues, np.full_like(hues, 0.62),
                                        np.full_like(hues, 0.92)], -1))
        lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
        if n:
            big = lab == int(np.argmax(ndimage.sum(occ, lab, range(1, n + 1)))) + 1
            img[big & ~ndimage.binary_erosion(big, structure=np.ones((3, 3)))] = 0.08
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


def finals_maxfill(root, L):
    """Final fill fraction s_max / L^2 per seed (dominant-component footprint)."""
    d = root / f"L{L}" / "runs"
    if not d.is_dir():
        return None
    v = [float(pd.read_csv(f)["max_size"].iloc[-1]) / (L * L)
         for f in sorted(d.glob("seed_*.csv"))]
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


def plabel(ax, s, dx=-0.20, dy=1.03):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="right", color="#1a1a1a")


def snap_axis(ax, title, accent=None):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for k, sp in ax.spines.items():
        sp.set_visible(True)
        sp.set_linewidth(1.1 if accent else 0.6)
        sp.set_color(accent if accent else "#b8b8b8")
    ax.set_title(title, fontsize=7, pad=2.5, color="#1a1a1a")


# ======================= FIGURE 2: phenomenon =======================
def figure_phenomenon():
    row_specs = [
        (200, 0, "(a)", "compact", [("full", 5000), ("full", 10000),
                                    ("zoom", 10000, (70, 130, 70, 130))]),
        (400, 8, "(b)", "runaway", [("full", 2000), ("full", 5000), ("full", 10000)]),
    ]
    coarse = {L: finals(OPEN, L) for L in range(200, 401, 20)}
    coarse = {L: v for L, v in coarse.items() if v is not None}
    fine = {L: finals(FINE, L) for L in range(300, 321, 2)}
    fine = {L: v for L, v in fine.items() if v is not None}

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.94))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 1.18],
                           hspace=0.52, wspace=0.46,
                           left=0.075, right=0.945, top=0.95, bottom=0.088)

    # (a,b) snapshots
    for row, (L, seed, lab, tag, spec) in enumerate(row_specs):
        for col, item in enumerate(spec):
            ax = fig.add_subplot(gs[row, col])
            cfg = load_cfg(L, seed, item[1])
            if cfg is not None:
                if item[0] == "zoom":
                    x0, x1, y0, y1 = item[2]
                    ax.imshow(render(cfg[x0:x1, y0:y1]), interpolation="nearest", origin="upper")
                    snap_axis(ax, f"$t={item[1]:,}$  (zoom {x1-x0}$\\times${y1-y0})", accent=C_OPEN)
                else:
                    ax.imshow(render(cfg), interpolation="nearest", origin="upper")
                    snap_axis(ax, f"$t={item[1]:,}$   max$\\,{maxcomp(cfg):,}$")
            if col == 0:
                ax.set_ylabel(f"$L={L}$\n({tag})", fontsize=8.5, fontweight="bold")
                plabel(ax, lab, dx=-0.14, dy=1.0)

    # (c) final mean size vs L: IQR band + median + per-run strip
    axc = fig.add_subplot(gs[2, 0])
    axc.axvspan(300, 320, color=BAND, zorder=0)
    cL = sorted(coarse)
    med = np.array([np.median(coarse[L]) for L in cL])
    q1 = np.array([np.percentile(coarse[L], 25) for L in cL])
    q3 = np.array([np.percentile(coarse[L], 75) for L in cL])
    axc.fill_between(cL, np.clip(q1, 1, None), np.clip(q3, 1, None),
                     color=C_OPEN, alpha=0.18, lw=0, zorder=1)
    for L, v in coarse.items():
        x = L + np.random.RandomState(L).uniform(-2.5, 2.5, size=len(v))
        axc.scatter(x, np.clip(v, 1, None), s=9, color=C_OPEN, alpha=0.5,
                    edgecolors="white", linewidths=0.25, zorder=3)
    for L, v in fine.items():
        x = L + np.random.RandomState(L).uniform(-0.8, 0.8, size=len(v))
        axc.scatter(x, np.clip(v, 1, None), s=8, color=C_PERI, alpha=0.6,
                    marker="D", edgecolors="white", linewidths=0.25, zorder=4)
    axc.plot(cL, np.clip(med, 1, None), color=C_MED, lw=1.6, zorder=5)
    axc.set_yscale("log"); axc.set_ylim(0.8, 2e5)
    axc.set_xlabel("Space size $L$"); axc.set_ylabel("Final mean size")
    axc.legend(handles=[
        Line2D([], [], color=C_MED, lw=1.6, label="Median"),
        Line2D([], [], marker="o", color=C_OPEN, lw=0, label=r"Coarse $\Delta L{=}20$"),
        Line2D([], [], marker="D", color=C_PERI, lw=0, label=r"Fine $\Delta L{=}2$"),
    ], loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.15)
    sns.despine(ax=axc); plabel(axc, "(c)")

    # (d) runaway fraction vs L with Wilson bands
    axd = fig.add_subplot(gs[2, 1])
    axd.axvspan(300, 320, color=BAND, zorder=0)
    for src, color, mk, lab in [(coarse, C_OPEN, "o", "Coarse"), (fine, C_PERI, "D", "Fine")]:
        Ls = sorted(src); fr = []; lo = []; hi = []
        for L in Ls:
            v = src[L]; k = int((v > RUN).sum()); n = len(v)
            fr.append(k / n); a, b = wilson(k, n); lo.append(a); hi.append(b)
        axd.fill_between(Ls, lo, hi, color=color, alpha=0.15, lw=0)
        axd.plot(Ls, fr, mk + "-", color=color, ms=4, lw=1.4, mec="white",
                 mew=0.4, label=lab)
    axd.set_ylim(-0.05, 1.05)
    axd.set_xlabel("Space size $L$"); axd.set_ylabel("Runaway fraction")
    axd.legend(loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.15)
    sns.despine(ax=axd); plabel(axd, "(d)")

    # (e) long-run novelty (L=200): patterns rise, size saturates
    axe = fig.add_subplot(gs[2, 2])
    files = sorted((LARGE / "L200" / "runs").glob("seed_*.csv"))[:40]
    dfs = [pd.read_csv(f)[["step", "mean_size", "cum_pattern_types"]] for f in files]
    steps = dfs[0]["step"].values
    pat = np.median(np.stack([d["cum_pattern_types"].values for d in dfs]), 0)
    sz = np.median(np.stack([d["mean_size"].values for d in dfs]), 0)
    l1, = axe.plot(steps, pat / 1000, color=C_PERI, lw=1.8)
    axe.set_xlabel("Time (steps)")
    axe.set_ylabel(r"Cum. pattern types ($10^3$)", color=C_PERI)
    axe.tick_params(axis="y", colors=C_PERI); axe.set_ylim(0, None)
    axe.spines["left"].set_color(C_PERI)
    ax2 = axe.twinx()
    l2, = ax2.plot(steps, sz, color=C_OPEN, lw=1.8)
    ax2.set_ylabel("Mean size", color=C_OPEN); ax2.tick_params(axis="y", colors=C_OPEN)
    ax2.set_ylim(0, max(6, sz.max() * 1.5)); ax2.spines["right"].set_color(C_OPEN)
    ax2.grid(False)
    axe.set_title(r"$L=200$: novelty persists, size saturates", fontsize=7.5)
    axe.legend(handles=[l1, l2], labels=["Cum. pattern types", "Mean size"],
               loc="center left", frameon=False, fontsize=6.8)
    sns.despine(ax=axe, right=False, top=True); plabel(axe, "(e)")

    out = FIG_DIR / "fig_schc_phenomenon"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    print(f"Saved {out}.pdf / .png")


# ======================= FIGURE 3: mechanism =======================
def phase_grid(param_col, param_vals, Ls, summ_csv):
    df = pd.read_csv(summ_csv)
    G = np.full((len(param_vals), len(Ls)), np.nan)
    for i, pv in enumerate(param_vals):
        for j, L in enumerate(Ls):
            row = df[(np.isclose(df[param_col], pv)) & (df["L"] == L)]
            if len(row):
                G[i, j] = float(row["mean_size_final"].iloc[0])
    return G


def annot(G):
    A = np.empty_like(G, dtype=object)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            v = G[i, j]
            A[i, j] = "" if np.isnan(v) else (f"{v:.0f}" if v < 1000 else f"{v/1000:.0f}k")
    return A


def draw_heat(ax, G, pvals, Ls, ptitle, ylabel, cbar_ax=None):
    # rows reversed so the smallest parameter value sits at the bottom
    Gr = G[::-1]; yl = [f"{p:g}" for p in pvals][::-1]
    hm = sns.heatmap(np.log10(np.clip(Gr, 1, None)), ax=ax, cmap="rocket",
                     vmin=0, vmax=5, annot=annot(Gr), fmt="", annot_kws={"size": 6.5},
                     linewidths=1.2, linecolor="white",
                     xticklabels=Ls, yticklabels=yl,
                     cbar=cbar_ax is not None, cbar_ax=cbar_ax,
                     cbar_kws={"label": r"$\log_{10}$ final mean size"})
    ax.set_xlabel("Space size $L$"); ax.set_ylabel(ylabel)
    ax.set_title(ptitle, fontsize=8)
    ax.tick_params(left=False, bottom=False)
    plt.setp(ax.get_yticklabels(), rotation=0)
    return hm


def figure_mechanism():
    bnd_L = [200, 240, 280, 300, 320, 360]
    ta = pd.read_csv(OPEN / "analysis" / "transition_analysis.csv")

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 1.02))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.05, 0.92],
                           hspace=0.62, wspace=0.36,
                           left=0.115, right=0.9, top=0.95, bottom=0.088)

    # (a) boundary control: dominant-component fill fraction s_max / L^2, per seed
    axa = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(42)
    for root, col, mk, lab, off, ls in [(OPEN, C_OPEN, "o", "Open", -5.0, "-"),
                                        (PERI, C_PERI, "D", "Periodic", 5.0, "--")]:
        meds = []
        for L in bnd_L:
            v = finals_maxfill(root, L)
            if v is None:
                meds.append(np.nan)
                continue
            xs = L + off + rng.uniform(-2.4, 2.4, size=v.size)
            axa.scatter(xs, v, s=13, color=col, marker=mk, alpha=0.5,
                        edgecolors="white", linewidths=0.3, zorder=3)
            meds.append(float(np.median(v)))
        axa.plot(bnd_L, meds, ls, color=col, lw=1.6, marker=mk, ms=4.5,
                 mec="white", mew=0.5, label=lab, zorder=4)
    axa.axhline(0.65, color="#9a9a9a", lw=0.8, ls=":")
    axa.text(bnd_L[0] - 4, 0.68, r"runaway fill $\approx 0.65$",
             fontsize=6.2, color="#8a8a8a")
    axa.set_ylim(-0.04, 0.82)
    axa.set_xlabel("Space size $L$")
    axa.set_ylabel(r"Final fill fraction $s_{\max}/L^{2}$")
    axa.legend(loc="center left", frameon=False, title="Boundary", title_fontsize=7)
    sns.despine(ax=axa); plabel(axa, "(a)")

    # (b) size-score correlation
    axb = fig.add_subplot(gs[0, 1])
    axb.axvspan(300, 320, color=BAND, zorder=0)
    axb.axhline(0, color="#9a9a9a", lw=0.7, ls=":")
    # 95% percentile-bootstrap intervals over runs (scripts/transition_uncertainty.py)
    unc = pd.read_csv(ROOT / "results" / "statistical_analysis" / "transition_uncertainty.csv")
    unc = unc.set_index("L").loc[ta["L"]]
    axb.errorbar(ta["L"], ta["corr_size_fitness_2k_10k"],
                 yerr=[ta["corr_size_fitness_2k_10k"].to_numpy() - unc["r_lo"].to_numpy(),
                       unc["r_hi"].to_numpy() - ta["corr_size_fitness_2k_10k"].to_numpy()],
                 fmt="none", ecolor=C_OPEN, alpha=0.45, elinewidth=1.0, capsize=2, zorder=2)
    axb.plot(ta["L"], ta["corr_size_fitness_2k_10k"], "o-", color=C_OPEN, ms=4,
             lw=1.4, mec="white", mew=0.4, zorder=3)
    axb.set_ylim(-1.0, 0.8)
    axb.set_xlabel("Space size $L$"); axb.set_ylabel(r"Corr($\bar s,\bar q$)")
    sns.despine(ax=axb); plabel(axb, "(b)")

    # (c,d) phase diagrams with a single shared colourbar
    L4 = [200, 300, 320, 400]
    cax = fig.add_axes([0.915, 0.40, 0.02, 0.22])
    axc = fig.add_subplot(gs[1, 0])
    Gmu = phase_grid("mu", [0.001, 0.002, 0.005, 0.01], L4,
                     ROOT / "results/param_sensitivity_mu/sensitivity_summary_mu.csv")
    draw_heat(axc, Gmu, [0.001, 0.002, 0.005, 0.01], L4,
              r"Final mean size vs $(L,\mu)$", r"Mutation rate $\mu$", cbar_ax=None)
    plabel(axc, "(c)", dx=-0.17)

    axd = fig.add_subplot(gs[1, 1])
    Gd = phase_grid("death_prob", [0.0005, 0.001, 0.005, 0.01], L4,
                    ROOT / "results/param_sensitivity_death/sensitivity_summary_death_prob.csv")
    draw_heat(axd, Gd, [0.0005, 0.001, 0.005, 0.01], L4,
              r"Final mean size vs $(L,p_{\mathrm{death}})$",
              r"Death prob. $p_{\mathrm{death}}$", cbar_ax=cax)
    plabel(axd, "(d)", dx=-0.17)

    # (e) nucleation kinetics
    axe = fig.add_subplot(gs[2, :])
    L_nuc = [320, 340, 360, 400]
    cols = sns.color_palette("flare", len(L_nuc))
    for idx, L in enumerate(L_nuc):
        d = OPEN / f"L{L}" / "runs"
        cross = []
        for f in sorted(d.glob("seed_*.csv")):
            df = pd.read_csv(f)
            over = df[df["mean_size"] > RUN]
            cross.append(int(over["step"].iloc[0]) if len(over) else np.inf)
        grid = np.linspace(0, 20000, 400)
        frac = [np.mean([c <= t for c in cross]) for t in grid]
        axe.plot(grid, frac, lw=2.0, color=cols[idx], label=f"$L={L}$", solid_capstyle="round")
    axe.set_xlabel("Time (steps)"); axe.set_ylabel("Fraction nucleated")
    axe.set_ylim(-0.03, 1.03); axe.set_xlim(0, 20000)
    leg = axe.legend(loc="upper left", frameon=False, ncol=2,
                     title="Nucleation kinetics", title_fontsize=7.5)
    leg._legend_box.align = "left"
    sns.despine(ax=axe); plabel(axe, "(e)", dx=-0.075)

    out = FIG_DIR / "fig_schc_mechanism"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    print(f"Saved {out}.pdf / .png")


if __name__ == "__main__":
    np.random.seed(0)
    figure_phenomenon()
    figure_mechanism()
