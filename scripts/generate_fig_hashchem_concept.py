#!/usr/bin/env python3
"""Conceptual schematic (Figure 1) explaining Hash Chemistry as a whole.

A hand-drawn "ponchi-e" style overview, four panels:
  (a) cardinality leap  : possibility space explodes with entity size
  (b) hash oracle       : a deterministic hash scores any structure in [0,1]
  (c) the engine        : compare -> higher score replicates -> mutation (loop)
  (d) embeddings        : the same rule in a non-spatial vs a spatial (SCHC) medium

Pure schematic (no data). Styled to match the seaborn figures.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.colors import hsv_to_rgb
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "tex" / "Large-Hash-Chemistry" / "archive" / "figures_unused"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(context="paper", style="ticks", font="DejaVu Sans")
plt.rcParams.update({"font.size": 8, "text.color": "#1a1a1a", "figure.dpi": 300,
                     "savefig.dpi": 300})
PAL = sns.color_palette("deep")
BLUE, RED, GREEN, PURPLE, GREY = PAL[0], PAL[3], PAL[2], PAL[4], "#5a5a5a"
INK = "#1a1a1a"
GOLDEN = 0.618033988749895
MM = 1 / 25.4
DOUBLE = 180 * MM


def tcol(t):
    return hsv_to_rgb(((t * GOLDEN) % 1.0, 0.62, 0.92))


def pattern(ax, cells, x0, y0, cell=0.05, outline=None, types=None):
    """Draw a small pattern of unit cells at (x0,y0) in axes coords."""
    types = types if types is not None else [i + 1 for i in range(len(cells))]
    for (dx, dy), t in zip(cells, types):
        ax.add_patch(Rectangle((x0 + dx * cell, y0 + dy * cell), cell, cell,
                               facecolor=tcol(t), edgecolor="white", lw=0.4, zorder=3))
    if outline:
        xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
        ax.add_patch(Rectangle((x0 + min(xs) * cell - cell * 0.25,
                                y0 + min(ys) * cell - cell * 0.25),
                               (max(xs) - min(xs) + 1.5) * cell,
                               (max(ys) - min(ys) + 1.5) * cell,
                               fill=False, edgecolor=outline, lw=1.3, zorder=4))


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=8, lw=1.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.006,rounding_size=0.03",
                 facecolor=fc, edgecolor=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=4,
            fontweight="bold" if bold else "normal")


def arrow(ax, p0, p1, color=INK, lw=1.4, style="-|>", ms=11, conn="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                 color=color, lw=lw, connectionstyle=conn, zorder=2))


def panel(ax, letter, title):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="right", color=INK)
    ax.text(0.5, 1.02, title, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom", ha="center", color=INK)


def score_bar(ax, x, y, w, h, val, label=None):
    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto",
              cmap="rocket", zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=INK, lw=1.0, zorder=4))
    mx = x + val * w
    ax.plot([mx, mx], [y - 0.012, y + h + 0.012], color=INK, lw=1.6, zorder=5)
    ax.text(x, y - 0.05, "0", ha="center", va="top", fontsize=6.5)
    ax.text(x + w, y - 0.05, "1", ha="center", va="top", fontsize=6.5)
    if label:
        ax.text(mx, y + h + 0.03, label, ha="center", va="bottom", fontsize=7,
                fontweight="bold")


# --------------------------------------------------------------------------
def main():
    fig = plt.figure(figsize=(DOUBLE, DOUBLE * 0.62))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.16,
                           left=0.04, right=0.985, top=0.9, bottom=0.03)

    # ---------- (a) cardinality leap ----------
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "(a)", "Cardinality leap")
    # axes arrows
    arrow(ax, (0.13, 0.12), (0.95, 0.12), color=GREY, lw=1.2)
    arrow(ax, (0.13, 0.12), (0.13, 0.92), color=GREY, lw=1.2)
    ax.text(0.55, 0.02, "entity size $n$", ha="center", fontsize=7, color=GREY)
    ax.text(0.03, 0.55, "possible\nstructures", ha="center", va="center",
            fontsize=7, color=GREY, rotation=90)
    xs = np.linspace(0.16, 0.9, 100)
    ys = 0.14 + 0.72 * (np.exp(3.0 * (xs - 0.16)) - 1) / (np.exp(3.0 * 0.74) - 1)
    ax.plot(xs, ys, color=BLUE, lw=2.2, zorder=2)
    # example structures of increasing size along the curve
    pattern(ax, [(0, 0)], 0.19, 0.20, cell=0.045)
    pattern(ax, [(0, 0), (1, 0)], 0.37, 0.30, cell=0.045)
    pattern(ax, [(0, 0), (1, 0), (0, 1), (1, 1)], 0.55, 0.48, cell=0.045)
    pattern(ax, [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
            0.7, 0.66, cell=0.042)
    ax.text(0.86, 0.86, r"$\sim k^{\,n}$", fontsize=9, color=BLUE, ha="center")
    ax.text(0.5, -0.06, "larger structures $\\Rightarrow$ vastly more variants",
            transform=ax.transAxes, ha="center", fontsize=6.8, color=GREY)

    # ---------- (b) hash oracle ----------
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "(b)", "Hash oracle")
    pattern(ax, [(0, 0), (1, 0), (1, 1), (2, 1)], 0.06, 0.5, cell=0.06,
            outline=INK)
    arrow(ax, (0.34, 0.6), (0.44, 0.6))
    box(ax, 0.44, 0.5, 0.24, 0.2, "hash\n$h(\\cdot)$", fc="#f2f2f2", bold=True, fs=8.5)
    arrow(ax, (0.68, 0.6), (0.78, 0.6))
    score_bar(ax, 0.5, 0.16, 0.42, 0.07, 0.72, label="score")
    ax.text(0.5, 0.9, "a deterministic hash maps\nany structure to a number in [0,1]",
            ha="center", va="center", fontsize=7, color=GREY)
    ax.text(0.2, 0.34, "structure", ha="center", fontsize=6.8, color=GREY)

    # ---------- (c) the engine ----------
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "(c)", "Compete $\\rightarrow$ replicate $\\rightarrow$ mutate")
    # two contestants
    pattern(ax, [(0, 0), (1, 0), (0, 1)], 0.08, 0.62, cell=0.05, outline=GREY)
    ax.text(0.16, 0.55, "$q_A=0.3$", ha="center", fontsize=7)
    pattern(ax, [(0, 0), (1, 0), (1, 1), (2, 1)], 0.08, 0.2, cell=0.05, outline=RED)
    ax.text(0.18, 0.13, "$q_B=0.8$", ha="center", fontsize=7, color=RED, fontweight="bold")
    box(ax, 0.36, 0.4, 0.2, 0.16, "compare", fc="#f2f2f2")
    arrow(ax, (0.31, 0.68), (0.37, 0.55), color=GREY)
    arrow(ax, (0.33, 0.28), (0.38, 0.42), color=RED)
    arrow(ax, (0.56, 0.48), (0.66, 0.48))
    ax.text(0.61, 0.53, "winner", fontsize=6.5, color=RED, ha="center")
    # winner replicates with a mutation (one changed cell = star/different colour)
    pattern(ax, [(0, 0), (1, 0), (1, 1)], 0.68, 0.4, cell=0.05,
            types=[3, 3, 3], outline=RED)
    pattern(ax, [(2, 1)], 0.68, 0.4, cell=0.05, types=[8])
    ax.text(0.66, 0.63, "replicate\n+ mutate", fontsize=6.8, color=RED, ha="left")
    ax.add_patch(Circle((0.68 + 2.5 * 0.05, 0.4 + 1.5 * 0.05), 0.022,
                        fill=False, edgecolor=INK, lw=1.1, zorder=6))
    ax.annotate("mutation", xy=(0.68 + 2.5 * 0.05, 0.4 + 1.5 * 0.05),
                xytext=(0.9, 0.22), fontsize=6.3, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=INK))
    # loop arrow back
    arrow(ax, (0.8, 0.36), (0.3, 0.16), color=GREY, style="-|>", lw=1.2,
          conn="arc3,rad=0.35")
    ax.text(0.5, 0.06, "repeat", fontsize=6.8, color=GREY, ha="center")

    # ---------- (d) embeddings ----------
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "(d)", "One rule, many media")
    # non-spatial multiset "bag"
    box(ax, 0.04, 0.2, 0.34, 0.55, "", fc="#f7f7f7", ec=GREY, lw=1.2)
    ax.text(0.21, 0.8, "non-spatial\n(multiset)", ha="center", fontsize=7, color=GREY)
    rng = np.random.RandomState(3)
    for _ in range(14):
        x = 0.08 + rng.uniform(0, 0.26); y = 0.26 + rng.uniform(0, 0.42)
        ax.add_patch(Rectangle((x, y), 0.03, 0.03, facecolor=tcol(rng.randint(1, 9)),
                               edgecolor="white", lw=0.3, zorder=3))
    # spatial SCHC grid
    gx, gy, n, cs = 0.55, 0.2, 8, 0.045
    ax.add_patch(Rectangle((gx, gy), n * cs, n * cs, fill=False, edgecolor=GREY, lw=1.0))
    comps = {(1, 1): 2, (2, 1): 2, (1, 2): 2,
             (5, 5): 4, (6, 5): 4, (6, 6): 4, (5, 6): 4,
             (3, 6): 6, (4, 6): 6}
    for (cxx, cyy), t in comps.items():
        ax.add_patch(Rectangle((gx + cxx * cs, gy + cyy * cs), cs, cs,
                               facecolor=tcol(t), edgecolor="white", lw=0.4, zorder=3))
    ax.text(0.55 + n * cs / 2, 0.8, "spatial: SCHC\n(grid components)", ha="center",
            fontsize=7, color=GREY)
    arrow(ax, (0.4, 0.47), (0.53, 0.47), color=INK, lw=1.4)
    ax.text(0.465, 0.52, "same\nengine", fontsize=6.3, color=INK, ha="center")

    fig.suptitle("Hash Chemistry: scoring structures of any size with a hash opens an unbounded possibility space",
                 fontsize=8.6, fontweight="bold", y=0.99)

    out = FIG_DIR / "fig_hashchem_concept"
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png")
    print(f"Saved {out}.pdf / .png")


if __name__ == "__main__":
    main()
