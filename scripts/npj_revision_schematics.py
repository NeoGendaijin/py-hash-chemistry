#!/usr/bin/env python3
"""Draw the revised Figure 1 schematic as editable vector artwork."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "results/npj_rev/schematics"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set(xlim=(0, 10), ylim=(0, 4.5), aspect="equal")
    ax.axis("off")
    colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#8B8B8B"]
    ax.add_patch(Rectangle((0.1, 0.45), 3.45, 3.45, fill=False, lw=1.2))
    particles = [(0.45, 1.0, 0), (0.7, 1.6, 2), (1.1, 2.3, 1),
                 (1.4, 3.25, 4), (1.8, 2.9, 2), (2.3, 3.4, 1),
                 (2.75, 2.95, 0), (3.1, 2.15, 3), (1.55, 1.2, 4),
                 (2.2, 1.45, 4), (2.5, 1.8, 1), (2.8, 1.5, 0)]
    for x, y, c in particles:
        ax.add_patch(Circle((x, y), 0.047, fc=colors[c], ec="white", lw=0.4))
    ax.add_patch(Circle((2.5, 1.55), 0.43, fill=False, ec="#757575", ls=":", lw=1.8))
    ax.text(1.82, 0.18, "Population of typed particles", ha="center", fontsize=10)
    ax.text(5.15, 4.0, "Higher-order entity", ha="center", fontsize=12)
    ax.add_patch(Circle((5.15, 2.85), 0.78, fill=False, ec="#757575", ls=":", lw=1.8))
    for x, y, c in [(4.85, 2.55, 4), (5.03, 3.2, 1), (5.55, 2.7, 0)]:
        ax.add_patch(Circle((x, y), 0.10, fc=colors[c], ec="white", lw=0.5))
    ax.add_patch(FancyArrowPatch((2.91, 1.82), (4.45, 2.42), arrowstyle="-|>",
                                mutation_scale=15, color="#757575", lw=1.6, ls=":"))
    ax.add_patch(FancyBboxPatch((7.05, 2.33), 2.65, 1.12, boxstyle="round,pad=0.10",
                               fc="#EEF3F5", ec="#49646F", lw=1.2))
    ax.text(8.375, 2.89, "Deterministic\nhash oracle", ha="center", va="center", fontsize=12)
    ax.add_patch(FancyArrowPatch((6.0, 2.85), (6.94, 2.85), arrowstyle="-|>",
                                mutation_scale=15, color="#49646F", lw=1.6))
    ax.add_patch(FancyArrowPatch((8.375, 2.2), (8.375, 1.65), arrowstyle="-|>",
                                mutation_scale=15, color="#49646F", lw=1.6))
    ax.text(8.375, 1.37, "Hash score", ha="center", va="center", fontsize=12)
    ax.text(5.25, 1.22, "Replication\nRemoval\nNo action", ha="left", va="center", fontsize=11,
            linespacing=1.5)
    ax.add_patch(FancyArrowPatch((7.65, 1.25), (6.5, 1.25), arrowstyle="-|>",
                                mutation_scale=15, color="#757575", lw=1.4, ls=":"))
    ax.add_patch(FancyArrowPatch((4.95, 1.25), (2.9, 1.38), arrowstyle="-|>",
                                mutation_scale=15, color="#757575", lw=1.4, ls=":"))
    fig.savefig(out / "outline-original.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out / "outline-original.svg", bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
