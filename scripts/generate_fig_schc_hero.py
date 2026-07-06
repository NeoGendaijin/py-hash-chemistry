#!/usr/bin/env python3
"""Build the consolidated SCHC case-study "hero" figure for the npj Complexity review.

A single multi-panel figure (a-f) with unified typography, redrawn from the raw
per-seed CSVs and saved snapshot configurations:

  (a) L=200 snapshots at t=2,000 / 5,000 / 10,000  (compact regime)
  (b) L=400 snapshots at t=2,000 / 5,000 / 10,000  (runaway regime)
  (c) final mean component size vs L (coarse Delta L=20 + fine Delta L=2), log
  (d) size-score temporal correlation vs L
  (e) runaway fraction vs L (open boundaries; Wilson 95% CI)
  (f) boundary-condition control: open vs periodic median final mean size (log)

Outputs tex/Large-Hash-Chemistry/figures/fig_schc_overview.{pdf,png}.
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

# ---- unified journal-style typography ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})
MM = 1 / 25.4
DOUBLE_COL = 180 * MM
C_OPEN = "#0072B2"      # blue
C_PERIODIC = "#D55E00"  # vermillion
C_MED = "#222222"
RUNAWAY_THRESHOLD = 100.0
GOLDEN = 0.618033988749895


def render_config(config_np):
    """RGB image: golden-ratio hue per cell type, black outline of largest component."""
    L = config_np.shape[0]
    img = np.ones((L, L, 3), dtype=np.float32)
    occ = config_np > 0
    if occ.any():
        types = config_np[occ].astype(np.float64)
        hues = (types * GOLDEN) % 1.0
        hsv = np.stack([hues, np.full_like(hues, 0.8), np.full_like(hues, 0.85)], axis=-1)
        img[occ] = hsv_to_rgb(hsv)
        labeled, n = ndimage.label(occ, structure=np.ones((3, 3)))
        if n:
            sizes = ndimage.sum(occ, labeled, range(1, n + 1))
            largest = labeled == (int(np.argmax(sizes)) + 1)
            interior = ndimage.binary_erosion(largest, structure=np.ones((3, 3)))
            img[largest & ~interior] = 0.0
    return img


def largest_comp_size(config_np):
    occ = config_np > 0
    if not occ.any():
        return 0
    labeled, n = ndimage.label(occ, structure=np.ones((3, 3)))
    if not n:
        return 0
    return int(ndimage.sum(occ, labeled, range(1, n + 1)).max())


def final_means(root: Path, L: int):
    runs = root / f"L{L}" / "runs"
    if not runs.is_dir():
        return None
    vals = []
    for f in sorted(runs.glob("seed_*.csv")):
        df = pd.read_csv(f)
        if len(df):
            vals.append(float(df["mean_size"].iloc[-1]))
    return np.array(vals, float) if vals else None


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def load_snapshots(subdir, steps):
    d = ROOT / "results" / "gifs" / subdir
    out = []
    for s in steps:
        p = d / f"config_step{s:05d}.npy"
        out.append(np.load(p) if p.exists() else None)
    return out


def main():
    STEPS = [2000, 5000, 10000]
    snaps_200 = load_snapshots("configs_L200_seed0", STEPS)
    snaps_400 = load_snapshots("configs_L400_seed8", STEPS)

    coarse_L = list(range(200, 401, 20))
    fine_L = list(range(300, 321, 2))
    OPEN = ROOT / "results" / "transition_scan"
    FINE = ROOT / "results" / "fine_transition_scan"
    PERI = ROOT / "results" / "boundary_control" / "periodic_v2"

    coarse = {L: final_means(OPEN, L) for L in coarse_L}
    coarse = {L: v for L, v in coarse.items() if v is not None}
    fine = {L: final_means(FINE, L) for L in fine_L}
    fine = {L: v for L, v in fine.items() if v is not None}

    ta = pd.read_csv(OPEN / "analysis" / "transition_analysis.csv")

    bnd_L = [200, 240, 280, 300, 320, 360]
    bnd_open = {L: final_means(OPEN, L) for L in bnd_L}
    bnd_peri = {L: final_means(PERI, L) for L in bnd_L}

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 1.18))
    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        height_ratios=[1.05, 1.05, 1.25, 1.25],
        hspace=0.55, wspace=0.34,
        left=0.085, right=0.985, top=0.955, bottom=0.065,
    )

    def panel_label(ax, s, dx=-0.14, dy=1.12):
        ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom", ha="right")

    # (a),(b) snapshots
    for row, (snaps, size, lab) in enumerate([(snaps_200, 200, "(a)"), (snaps_400, 400, "(b)")]):
        for col, (cfg, t) in enumerate(zip(snaps, STEPS)):
            ax = fig.add_subplot(gs[row, col])
            if cfg is not None:
                ax.imshow(render_config(cfg), interpolation="nearest", origin="upper")
                mc = largest_comp_size(cfg)
                ax.set_title(f"$t={t:,}$   (max {mc:,})", fontsize=7, pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color("0.6")
            if col == 0:
                ax.set_ylabel(f"$L={size}$", fontsize=8, fontweight="bold")
                panel_label(ax, lab, dx=-0.10)

    # (c) final mean size vs L
    axc = fig.add_subplot(gs[2, 0])
    axc.axvspan(300, 320, color="0.88", zorder=0)
    for L, v in coarse.items():
        axc.scatter(np.full_like(v, L), np.clip(v, 1, None), s=6, color=C_OPEN,
                    alpha=0.35, edgecolors="none", zorder=2)
    cm_L = sorted(coarse); cm_med = [np.median(coarse[L]) for L in cm_L]
    axc.plot(cm_L, cm_med, "-", color=C_MED, lw=1.2, zorder=4, label="Median")
    for L, v in fine.items():
        axc.scatter(np.full_like(v, L), np.clip(v, 1, None), s=5, color=C_PERIODIC,
                    alpha=0.5, marker="s", edgecolors="none", zorder=3)
    axc.set_yscale("log"); axc.set_xlabel("Space size $L$")
    axc.set_ylabel("Final mean size")
    axc.set_ylim(0.8, 2e5)
    axc.scatter([], [], s=6, color=C_OPEN, label=r"Coarse ($\Delta L{=}20$)")
    axc.scatter([], [], s=5, color=C_PERIODIC, marker="s", label=r"Fine ($\Delta L{=}2$)")
    axc.legend(loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.2)
    panel_label(axc, "(c)")

    # (d) size-score correlation vs L
    axd = fig.add_subplot(gs[2, 1])
    axd.axvspan(300, 320, color="0.88", zorder=0)
    axd.axhline(0, color="0.6", lw=0.6, ls=":")
    axd.plot(ta["L"], ta["corr_size_fitness_2k_10k"], "o-", color=C_OPEN, ms=3.5, lw=1.0)
    axd.set_xlabel("Space size $L$"); axd.set_ylabel(r"Corr($\bar s,\bar q$)")
    axd.set_ylim(-1.0, 0.5)
    panel_label(axd, "(d)")

    # (e) runaway fraction vs L (open) with Wilson CI
    axe = fig.add_subplot(gs[2, 2])
    axe.axvspan(300, 320, color="0.88", zorder=0)
    for src, color, marker, lab in [(coarse, C_OPEN, "o", r"Coarse"),
                                    (fine, C_PERIODIC, "s", r"Fine")]:
        Ls = sorted(src); fr = []; lo = []; hi = []
        for L in Ls:
            v = src[L]; k = int((v > RUNAWAY_THRESHOLD).sum()); n = len(v)
            fr.append(k / n); a, b = wilson(k, n); lo.append(fr[-1] - a); hi.append(b - fr[-1])
        axe.errorbar(Ls, fr, yerr=[lo, hi], fmt=marker + "-", color=color, ms=3.5,
                     lw=1.0, capsize=1.5, elinewidth=0.7, label=lab)
    axe.set_xlabel("Space size $L$"); axe.set_ylabel("Runaway fraction")
    axe.set_ylim(-0.05, 1.05)
    axe.legend(loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.2)
    panel_label(axe, "(e)")

    # (f) boundary control: open vs periodic median final mean size
    axf = fig.add_subplot(gs[3, :])
    Ls = [L for L in bnd_L if bnd_open.get(L) is not None]
    om = [np.median(bnd_open[L]) for L in Ls]
    pm = [np.median(bnd_peri[L]) if bnd_peri.get(L) is not None else np.nan for L in Ls]
    axf.plot(Ls, om, "o-", color=C_OPEN, ms=5, lw=1.4, label="Open boundary")
    axf.plot(Ls, pm, "s--", color=C_PERIODIC, ms=5, lw=1.4, label="Periodic (toroidal)")
    axf.axhline(RUNAWAY_THRESHOLD, color="0.6", lw=0.7, ls=":")
    axf.text(Ls[0], RUNAWAY_THRESHOLD * 1.25, "runaway threshold", fontsize=6.5, color="0.4")
    axf.set_yscale("log"); axf.set_xlabel("Space size $L$")
    axf.set_ylabel("Median final mean size")
    axf.set_ylim(1, 5e3)
    axf.legend(loc="center right", frameon=False, title="Boundary condition")
    panel_label(axf, "(f)", dx=-0.06)

    out = FIG_DIR / "fig_schc_overview"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=300)
    print(f"Saved {out}.pdf / .png")


if __name__ == "__main__":
    main()
