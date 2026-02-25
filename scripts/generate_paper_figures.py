#!/usr/bin/env python3
"""Generate all publication-quality figures for the SCHC paper.

Style: Nature / npj Complexity
- Font: Arial, min 7pt
- DPI: 300
- Colors: colorblind-friendly (seaborn "colorblind")
- Line width: 1.0-1.5pt, axes 0.5pt
- Error bands: shaded alpha=0.2-0.3
- Panel labels: bold lowercase (a), (b), ... top-left
- No titles on panels
- White background, minimal chartjunk
- Single column = 88mm, double column = 180mm
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.ticker as ticker
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "tex", "Large-Hash-Chemistry", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
MM = 1 / 25.4  # mm → inches
SINGLE_COL = 88 * MM
DOUBLE_COL = 180 * MM

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
})

PALETTE = sns.color_palette("colorblind", 10)


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add bold lowercase panel label, e.g. '(a)'."""
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def save(fig, name):
    """Save as both PDF and PNG."""
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), format="pdf")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), format="png", dpi=300)
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Grid snapshots (L=200 and L=400 composites)
# ═══════════════════════════════════════════════════════════════════════
def figure1():
    print("Figure 1: Grid snapshots")
    snap_dir = os.path.join(RESULTS, "snapshots")
    img200 = imread(os.path.join(snap_dir, "L200", "composite_L200.png"))
    img400 = imread(os.path.join(snap_dir, "L400", "composite_L400.png"))

    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COL, DOUBLE_COL * 1.2))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].imshow(img200)
    axes[1].imshow(img400)
    add_panel_label(axes[0], "a", x=-0.02, y=1.05)
    add_panel_label(axes[1], "b", x=-0.02, y=1.05)
    fig.subplots_adjust(hspace=0.05)
    save(fig, "fig1_snapshots")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Long-run dynamics comparison (L=100, 200, 400)
# ═══════════════════════════════════════════════════════════════════════
def figure2():
    print("Figure 2: Long-run dynamics")
    ls_dir = os.path.join(RESULTS, "large_space")
    sizes = [100, 200, 400]
    colors = [PALETTE[0], PALETTE[1], PALETTE[2]]
    labels = [f"$L={s}$" for s in sizes]

    data = {}
    for L in sizes:
        path = os.path.join(ls_dir, f"L{L}", "summary.csv")
        data[L] = pd.read_csv(path)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.6))
    metrics = [
        ("mean_fitness_mean", "mean_fitness_std", "Mean fitness"),
        ("max_fitness_mean", "max_fitness_std", "Max fitness"),
        ("mean_size_mean", "mean_size_std", "Mean component size"),
        ("cum_pattern_types_mean", "cum_pattern_types_std", "Cumulative pattern types"),
    ]
    panel_labels = ["a", "b", "c", "d"]

    for idx, (ax, (m, s, ylabel), pl) in enumerate(zip(axes.flat, metrics, panel_labels)):
        for L, c, lb in zip(sizes, colors, labels):
            df = data[L]
            steps = df["step"].values
            mask = steps > 0  # skip step 0 for log scale
            x = steps[mask]
            y = df[m].values[mask]
            ys = df[s].values[mask]
            ax.plot(x, y, color=c, label=lb, linewidth=1.2)
            ax.fill_between(x, y - ys, y + ys, color=c, alpha=0.2)
        ax.set_xscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        add_panel_label(ax, pl)
        if idx == 2:
            ax.set_yscale("log")
        if idx == 3:
            ax.set_yscale("log")

    axes[0, 0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "fig2_dynamics")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Phase transition — size and fitness vs L
# ═══════════════════════════════════════════════════════════════════════
def figure3():
    print("Figure 3: Phase transition scan")
    # Load coarse scan
    coarse = pd.read_csv(os.path.join(RESULTS, "transition_scan", "analysis", "transition_analysis.csv"))
    # Load fine scan
    fine = pd.read_csv(os.path.join(RESULTS, "fine_transition_scan", "analysis", "fine_transition_analysis.csv"))
    # Load per-size summary for error bars
    pss = pd.read_csv(os.path.join(RESULTS, "statistical_analysis", "per_size_summary.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_COL * 0.65))

    # ── Panel (a): mean size at final step vs L (log y) ──
    ax = axes[0]
    # Coarse scan with error bars from pss
    ax.errorbar(pss["L"], pss["mean_size_final_mean"], yerr=pss["mean_size_final_std"],
                fmt="o-", color=PALETTE[0], markersize=4, linewidth=1.0,
                capsize=2, label="Coarse ($\\Delta L=20$)")
    # Fine scan
    ax.errorbar(fine["L"], fine["mean_size_end"], yerr=fine["mean_size_end_std"],
                fmt="s--", color=PALETTE[1], markersize=3.5, linewidth=0.8,
                capsize=2, label="Fine ($\\Delta L=2$)")
    ax.set_yscale("log")
    ax.set_ylabel("Mean size (final step)")
    ax.set_xlabel("System size $L$")
    ax.axvspan(300, 320, alpha=0.1, color="grey", label="Transition region")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    add_panel_label(ax, "a")

    # ── Panel (b): correlation(size, fitness) vs L ──
    ax = axes[1]
    ax.plot(coarse["L"], coarse["corr_size_fitness_2k_10k"], "o-", color=PALETTE[0],
            markersize=4, label="Coarse")
    ax.plot(fine["L"], fine["corr_size_fitness_2k_10k"], "s--", color=PALETTE[1],
            markersize=3.5, label="Fine")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.axvspan(300, 320, alpha=0.1, color="grey")
    ax.set_ylabel("Corr(size, fitness)")
    ax.set_xlabel("System size $L$")
    ax.legend(loc="lower left", frameon=False, fontsize=6)
    add_panel_label(ax, "b")

    # ── Panel (c): runaway fraction vs L ──
    ax = axes[2]
    ax.plot(pss["L"], pss["runaway_fraction"], "o-", color=PALETTE[0],
            markersize=4, label="Coarse")
    ax.plot(fine["L"], fine["runaway_fraction"], "s--", color=PALETTE[1],
            markersize=3.5, label="Fine")
    ax.axvspan(300, 320, alpha=0.1, color="grey")
    ax.set_ylabel("Runaway fraction")
    ax.set_xlabel("System size $L$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    add_panel_label(ax, "c")

    fig.tight_layout()
    save(fig, "fig3_transition")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Fine-grained transition (L=300–320) time series
# ═══════════════════════════════════════════════════════════════════════
def figure4():
    print("Figure 4: Fine transition time series")
    fine_dir = os.path.join(RESULTS, "fine_transition_scan")
    L_vals = list(range(300, 322, 2))  # 300, 302, ..., 320
    n = len(L_vals)
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (n - 1)) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.65))

    for i, L in enumerate(L_vals):
        df = pd.read_csv(os.path.join(fine_dir, f"L{L}", "summary.csv"))
        steps = df["step"].values
        mask = steps > 0
        x = steps[mask]

        # Mean size
        y = df["mean_size_mean"].values[mask]
        ys = df["mean_size_std"].values[mask]
        axes[0].plot(x, y, color=colors[i], linewidth=0.9, label=f"{L}")
        axes[0].fill_between(x, np.maximum(y - ys, 0.1), y + ys,
                             color=colors[i], alpha=0.15)

        # Mean fitness
        y2 = df["mean_fitness_mean"].values[mask]
        ys2 = df["mean_fitness_std"].values[mask]
        axes[1].plot(x, y2, color=colors[i], linewidth=0.9, label=f"{L}")
        axes[1].fill_between(x, y2 - ys2, y2 + ys2,
                             color=colors[i], alpha=0.15)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean component size")
    add_panel_label(axes[0], "a")

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean fitness")
    add_panel_label(axes[1], "b")

    # Colorbar-like legend
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=300, vmax=320))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("$L$", fontsize=8)
    save(fig, "fig4_fine_transition")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: Time to threshold crossing
# ═══════════════════════════════════════════════════════════════════════
def figure5():
    print("Figure 5: Time to threshold crossing")
    # Compute from individual runs in transition_scan
    ts_dir = os.path.join(RESULTS, "transition_scan")
    L_values = list(range(200, 420, 20))
    thresholds = [10, 100, 1000]

    # Collect per-run crossing times
    crossing_data = {thr: [] for thr in thresholds}
    for L in L_values:
        runs_dir = os.path.join(ts_dir, f"L{L}", "runs")
        seed_files = sorted(f for f in os.listdir(runs_dir) if f.endswith(".csv"))
        for sf in seed_files:
            df = pd.read_csv(os.path.join(runs_dir, sf))
            for thr in thresholds:
                crossed = df[df["mean_size"] >= thr]
                if len(crossed) > 0:
                    crossing_data[thr].append({"L": L, "time": crossed["step"].iloc[0]})

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_COL * 0.6))
    thr_colors = [PALETTE[0], PALETTE[2], PALETTE[3]]

    for idx, (thr, ax, c) in enumerate(zip(thresholds, axes, thr_colors)):
        cdf = pd.DataFrame(crossing_data[thr])
        if len(cdf) == 0:
            ax.text(0.5, 0.5, "No crossings", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="grey")
        else:
            # Only plot L values that have at least 1 crossing
            L_with_data = sorted(cdf["L"].unique())
            positions = []
            data_groups = []
            for L in L_with_data:
                vals = cdf[cdf["L"] == L]["time"].values
                if len(vals) >= 1:
                    positions.append(L)
                    data_groups.append(vals)

            if data_groups:
                vp = ax.violinplot(data_groups, positions=positions,
                                   widths=15, showmeans=True, showmedians=False)
                for body in vp["bodies"]:
                    body.set_facecolor(c)
                    body.set_alpha(0.5)
                vp["cmeans"].set_color(c)
                vp["cmins"].set_color(c)
                vp["cmaxes"].set_color(c)
                vp["cbars"].set_color(c)

                # Add individual points
                for L, vals in zip(positions, data_groups):
                    jitter = np.random.default_rng(42).uniform(-4, 4, len(vals))
                    ax.scatter(L + jitter, vals, s=8, color=c, alpha=0.6, zorder=3)

        ax.axvspan(300, 320, alpha=0.1, color="grey")
        ax.set_xlabel("System size $L$")
        if idx == 0:
            ax.set_ylabel("Crossing time (steps)")
        ax.set_xlim(190, 410)
        label_map = {10: "\\bar{s} \\geq 10", 100: "\\bar{s} \\geq 100", 1000: "\\bar{s} \\geq 1000"}
        ax.text(0.95, 0.95, f"${label_map[thr]}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8))
        add_panel_label(ax, chr(ord("a") + idx))

    fig.tight_layout()
    save(fig, "fig5_crossing_times")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: Parameter sensitivity (mu)
# ═══════════════════════════════════════════════════════════════════════
def figure6():
    print("Figure 6: Mutation rate sensitivity")
    mu_dir = os.path.join(RESULTS, "param_sensitivity_mu")
    mu_vals = [0.001, 0.002, 0.005, 0.01]
    L_vals = [200, 300, 320, 400]
    mu_colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]

    # Read final-step summary from each condition's summary.csv
    mu_data = {}  # (mu, L) -> last row of summary
    for mu in mu_vals:
        mu_str = f"{mu:.4f}"
        for L in L_vals:
            path = os.path.join(mu_dir, f"mu_{mu_str}", f"L{L}", "summary.csv")
            df = pd.read_csv(path)
            mu_data[(mu, L)] = df.iloc[-1]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))

    # ── Panel (a): mean size at final step vs L for each mu ──
    ax = axes[0, 0]
    for mu, c in zip(mu_vals, mu_colors):
        xs = L_vals
        ys = [mu_data[(mu, L)]["mean_size_mean"] for L in L_vals]
        yerr = [mu_data[(mu, L)]["mean_size_std"] for L in L_vals]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=c, markersize=4,
                    capsize=2, linewidth=1.0, label=f"$\\mu={mu}$")
    ax.set_yscale("log")
    ax.set_xlabel("System size $L$")
    ax.set_ylabel("Mean size (final step)")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    add_panel_label(ax, "a")

    # ── Panel (b): max size at final step vs L for each mu ──
    ax = axes[0, 1]
    for mu, c in zip(mu_vals, mu_colors):
        xs = L_vals
        ys = [mu_data[(mu, L)]["max_size_mean"] for L in L_vals]
        yerr = [mu_data[(mu, L)]["max_size_std"] for L in L_vals]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=c, markersize=4,
                    capsize=2, linewidth=1.0, label=f"$\\mu={mu}$")
    ax.set_yscale("log")
    ax.set_xlabel("System size $L$")
    ax.set_ylabel("Max size (final step)")
    add_panel_label(ax, "b")

    # ── Panel (c): mean fitness at final step vs L ──
    ax = axes[1, 0]
    for mu, c in zip(mu_vals, mu_colors):
        xs = L_vals
        ys = [mu_data[(mu, L)]["mean_fitness_mean"] for L in L_vals]
        yerr = [mu_data[(mu, L)]["mean_fitness_std"] for L in L_vals]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=c, markersize=4,
                    capsize=2, linewidth=1.0, label=f"$\\mu={mu}$")
    ax.set_xlabel("System size $L$")
    ax.set_ylabel("Mean fitness (final step)")
    ax.legend(loc="lower left", frameon=False, fontsize=6)
    add_panel_label(ax, "c")

    # ── Panel (d): time series of mean size for L=400 across mu values ──
    ax = axes[1, 1]
    for mu, c in zip(mu_vals, mu_colors):
        mu_str = f"{mu:.4f}"
        path = os.path.join(mu_dir, f"mu_{mu_str}", "L400", "summary.csv")
        df = pd.read_csv(path)
        mask = df["step"] > 0
        x = df["step"].values[mask]
        y = df["mean_size_mean"].values[mask]
        ys = df["mean_size_std"].values[mask]
        ax.plot(x, y, color=c, linewidth=1.0, label=f"$\\mu={mu}$")
        ax.fill_between(x, np.maximum(y - ys, 0.1), y + ys, color=c, alpha=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean size ($L=400$)")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    add_panel_label(ax, "d")

    fig.tight_layout()
    save(fig, "fig6_mu_sensitivity")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7: Parameter sensitivity (death_prob)
# ═══════════════════════════════════════════════════════════════════════
def figure7():
    print("Figure 7: Death probability sensitivity")
    dp_dir = os.path.join(RESULTS, "param_sensitivity_death")
    dp_vals = [0.0005, 0.001, 0.005, 0.01]
    L_vals = [200, 300, 320, 400]
    dp_colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]

    # Read final-step summary from each condition's summary.csv
    dp_data = {}
    for dp in dp_vals:
        dp_str = f"{dp:.4f}"
        for L in L_vals:
            path = os.path.join(dp_dir, f"death_prob_{dp_str}", f"L{L}", "summary.csv")
            df = pd.read_csv(path)
            dp_data[(dp, L)] = df.iloc[-1]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))

    # ── Panel (a): mean size at final step vs L for each death_prob ──
    ax = axes[0, 0]
    for dp, c in zip(dp_vals, dp_colors):
        xs = L_vals
        ys = [dp_data[(dp, L)]["mean_size_mean"] for L in L_vals]
        yerr = [dp_data[(dp, L)]["mean_size_std"] for L in L_vals]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=c, markersize=4,
                    capsize=2, linewidth=1.0, label=f"$p_{{\\mathrm{{death}}}}={dp}$")
    ax.set_yscale("log")
    ax.set_xlabel("System size $L$")
    ax.set_ylabel("Mean size (final step)")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    add_panel_label(ax, "a")

    # ── Panel (b): max size at final step vs L for each death_prob ──
    ax = axes[0, 1]
    for dp, c in zip(dp_vals, dp_colors):
        xs = L_vals
        ys = [dp_data[(dp, L)]["max_size_mean"] for L in L_vals]
        yerr = [dp_data[(dp, L)]["max_size_std"] for L in L_vals]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=c, markersize=4,
                    capsize=2, linewidth=1.0, label=f"$p_{{\\mathrm{{death}}}}={dp}$")
    ax.set_yscale("log")
    ax.set_xlabel("System size $L$")
    ax.set_ylabel("Max size (final step)")
    add_panel_label(ax, "b")

    # ── Panel (c): mean fitness at final step vs L ──
    ax = axes[1, 0]
    for dp, c in zip(dp_vals, dp_colors):
        xs = L_vals
        ys = [dp_data[(dp, L)]["mean_fitness_mean"] for L in L_vals]
        yerr = [dp_data[(dp, L)]["mean_fitness_std"] for L in L_vals]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=c, markersize=4,
                    capsize=2, linewidth=1.0, label=f"$p_{{\\mathrm{{death}}}}={dp}$")
    ax.set_xlabel("System size $L$")
    ax.set_ylabel("Mean fitness (final step)")
    ax.legend(loc="lower left", frameon=False, fontsize=6)
    add_panel_label(ax, "c")

    # ── Panel (d): time series of mean size for L=320 across death_prob values ──
    ax = axes[1, 1]
    for dp, c in zip(dp_vals, dp_colors):
        dp_str = f"{dp:.4f}"
        path = os.path.join(dp_dir, f"death_prob_{dp_str}", "L320", "summary.csv")
        df = pd.read_csv(path)
        mask = df["step"] > 0
        x = df["step"].values[mask]
        y = df["mean_size_mean"].values[mask]
        ys = df["mean_size_std"].values[mask]
        ax.plot(x, y, color=c, linewidth=1.0, label=f"$p_{{\\mathrm{{death}}}}={dp}$")
        ax.fill_between(x, np.maximum(y - ys, 0.1), y + ys, color=c, alpha=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean size ($L=320$)")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    add_panel_label(ax, "d")

    fig.tight_layout()
    save(fig, "fig7_death_sensitivity")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 8: Alternative hash function comparison
# ═══════════════════════════════════════════════════════════════════════
def figure8():
    print("Figure 8: Alternative hash (CRC32) comparison")
    alt_dir = os.path.join(RESULTS, "alt_hash")
    # Default hash data comes from death_prob=0.001 sensitivity (same params, same seeds)
    default_dir = os.path.join(RESULTS, "param_sensitivity_death", "death_prob_0.0010")
    L_vals = [200, 300, 320]  # L=400 has only 3 runs for alt_hash

    # ── Load alt_hash summary data ──
    alt_data = {}
    for L in L_vals:
        path = os.path.join(alt_dir, f"L{L}", "summary.csv")
        df = pd.read_csv(path)
        alt_data[L] = df

    # Load alt_hash L=400 individual runs (only 3 seeds)
    alt_L400_runs = []
    for seed in range(3):
        path = os.path.join(alt_dir, "L400", "runs", f"seed_{seed}.csv")
        alt_L400_runs.append(pd.read_csv(path))
    # Compute mean across 3 runs
    alt_L400_mean_size = np.mean([r["mean_size"].iloc[-1] for r in alt_L400_runs])
    alt_L400_mean_size_std = np.std([r["mean_size"].iloc[-1] for r in alt_L400_runs])
    alt_L400_fitness = np.mean([r["mean_fitness"].iloc[-1] for r in alt_L400_runs])
    alt_L400_fitness_std = np.std([r["mean_fitness"].iloc[-1] for r in alt_L400_runs])

    # ── Load default hash summary data ──
    def_data = {}
    for L in L_vals + [400]:
        path = os.path.join(default_dir, f"L{L}", "summary.csv")
        df = pd.read_csv(path)
        def_data[L] = df

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.65))

    # ── Panel (a): Mean size at final step vs L (both hash functions) ──
    ax = axes[0]
    all_L = [200, 300, 320, 400]

    # Default hash
    def_sizes = [def_data[L].iloc[-1]["mean_size_mean"] for L in all_L]
    def_sizes_err = [def_data[L].iloc[-1]["mean_size_std"] for L in all_L]
    ax.errorbar([x - 4 for x in all_L], def_sizes, yerr=def_sizes_err,
                fmt="o-", color=PALETTE[0], markersize=5, capsize=3,
                linewidth=1.2, label="Default (FNV-XOR)")

    # Alt hash
    alt_sizes = [alt_data[L].iloc[-1]["mean_size_mean"] for L in L_vals] + [alt_L400_mean_size]
    alt_sizes_err = [alt_data[L].iloc[-1]["mean_size_std"] for L in L_vals] + [alt_L400_mean_size_std]
    ax.errorbar([x + 4 for x in all_L], alt_sizes, yerr=alt_sizes_err,
                fmt="s--", color=PALETTE[2], markersize=5, capsize=3,
                linewidth=1.2, label="CRC32 observation")

    ax.set_yscale("log")
    ax.set_xlabel("System size $L$")
    ax.set_ylabel("Mean size (final step)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    add_panel_label(ax, "a")

    # ── Panel (b): Time series at L=320, comparing fitness from both hashes ──
    ax = axes[1]
    def_df = def_data[320]
    alt_df = alt_data[320]
    mask = def_df["step"] > 0

    # Default hash fitness
    x = def_df["step"].values[mask]
    y_def = def_df["mean_fitness_mean"].values[mask]
    ys_def = def_df["mean_fitness_std"].values[mask]
    ax.plot(x, y_def, color=PALETTE[0], linewidth=1.2, label="FNV-XOR fitness")
    ax.fill_between(x, y_def - ys_def, y_def + ys_def, color=PALETTE[0], alpha=0.2)

    # CRC32 fitness
    y_alt = alt_df["mean_fitness_mean"].values[mask]
    ys_alt = alt_df["mean_fitness_std"].values[mask]
    ax.plot(x, y_alt, color=PALETTE[2], linewidth=1.2, label="CRC32 fitness")
    ax.fill_between(x, y_alt - ys_alt, y_alt + ys_alt, color=PALETTE[2], alpha=0.2)

    ax.set_xscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean fitness ($L=320$)")
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    add_panel_label(ax, "b")

    fig.tight_layout()
    save(fig, "fig8_alt_hash")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Saving figures to: {FIG_DIR}\n")
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    print("\nAll figures generated.")
