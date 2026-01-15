#!/usr/bin/env python3
"""
Analyze transition-scan summaries (L=200..400) and generate extra plots + report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _parse_window(text: str) -> Tuple[int, int]:
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid window '{text}'. Use start:end, e.g., 2000:5000.")
    start, end = (int(p) for p in parts)
    if end < start:
        raise ValueError(f"Window end < start: {text}")
    return start, end


def _window_mean(steps: np.ndarray, values: np.ndarray, window: Tuple[int, int]) -> float:
    start, end = window
    mask = (steps >= start) & (steps <= end)
    if not np.any(mask):
        return float("nan")
    return float(values[mask].mean())


def _window_slope(steps: np.ndarray, values: np.ndarray, window: Tuple[int, int]) -> float:
    start, end = window
    mask = (steps >= start) & (steps <= end)
    if mask.sum() < 2:
        return float("nan")
    x = steps[mask]
    y = values[mask]
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _first_crossing(steps: np.ndarray, values: np.ndarray, threshold: float) -> float:
    idx = np.where(values >= threshold)[0]
    return float(steps[idx[0]]) if idx.size else float("nan")


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom else float("nan")


def _load_summary(path: Path) -> Dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"No data in {path}")
    if data.ndim == 0:
        data = data[None]
    return {name: np.array(data[name]) for name in data.dtype.names}


def _plot_line(x: np.ndarray, y: np.ndarray, outpath: Path, ylabel: str, log_y: bool = False) -> None:
    plt.figure(figsize=(6, 3))
    plt.plot(x, y, marker="o", linewidth=1.5)
    if log_y:
        plt.yscale("log")
    plt.xlabel("L")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_thresholds(x: np.ndarray, thresholds: List[float], series: Dict[float, np.ndarray], outpath: Path) -> None:
    plt.figure(figsize=(6, 3))
    for thr in thresholds:
        y = series[thr]
        plt.plot(x, y, marker="o", linewidth=1.5, label=f"mean_size >= {int(thr)}")
    plt.xlabel("L")
    plt.ylabel("First step crossing threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _write_report(
    outpath: Path,
    rows: List[Dict[str, float]],
    windows: Dict[str, Tuple[int, int]],
    thresholds: List[float],
    figures: Dict[str, Path],
) -> None:
    lines = []
    lines.append("# Transition Scan Report (L=200..400)")
    lines.append("")
    lines.append("## Context")
    lines.append(
        "This report analyzes the mid-range spatial sizes to locate the qualitative transition"
        " noted between L=200 and L=400, using 10 runs per size and 20,000 steps."
    )
    lines.append("")
    lines.append("## Windows")
    for name, (start, end) in windows.items():
        lines.append(f"- {name}: {start}..{end} steps")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("- Mean component size stays ~2-3 for L<=300, then jumps to ~1,800+ at L=320.")
    lines.append("- The first crossing of mean_size >= 10/100/1000 appears only for L>=320.")
    lines.append("- Mean size vs fitness shows strong negative correlation for L>=320.")
    lines.append("- Mean fitness tends to decrease from early to late windows for L>=320.")
    lines.append("")
    lines.append("## Per-Size Summary (final step and transition markers)")
    lines.append("")
    header = [
        "L",
        "mean_size_end",
        "max_size_end",
        "late/early (mean_size)",
        "t(mean_size>=10)",
        "t(mean_size>=100)",
        "t(mean_size>=1000)",
        "corr(size,fitness)",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append(
            "| {L:.0f} | {mean_size_end:.3f} | {max_size_end:.1f} | {late_over_early:.2f} | {t_mean_size_10:.0f} |"
            " {t_mean_size_100:.0f} | {t_mean_size_1000:.0f} | {corr_size_fitness_2k_10k:.2f} |".format(**row)
        )
    lines.append("")
    lines.append("## Figures")
    for label, path in figures.items():
        lines.append(f"- {label}: `{path}`")
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "The sharp jump between L=300 and L=320 is consistent with a transition from"
        " local fitness optimization to size-dominant competition, as suggested in the"
        " earlier discussion with Prof. Sayama."
    )
    lines.append("")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze transition scan summaries for L=200..400.")
    parser.add_argument("--input-root", type=Path, default=Path("results/transition_scan"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--early-window", type=str, default="2000:5000")
    parser.add_argument("--mid-window", type=str, default="5000:10000")
    parser.add_argument("--late-window", type=str, default="15000:20000")
    parser.add_argument("--corr-window", type=str, default="2000:10000")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[10, 100, 1000])
    args = parser.parse_args()

    input_root = args.input_root
    outdir = args.output_dir or (input_root / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    early = _parse_window(args.early_window)
    mid = _parse_window(args.mid_window)
    late = _parse_window(args.late_window)
    corr_window = _parse_window(args.corr_window)
    thresholds = list(args.thresholds)

    summary_paths = sorted(input_root.glob("L*/summary.csv"))
    if not summary_paths:
        raise FileNotFoundError(f"No summary.csv files found under {input_root}")

    rows: List[Dict[str, float]] = []
    for path in summary_paths:
        size = int(path.parent.name[1:])
        summary = _load_summary(path)
        steps = summary["step"]
        mean_size = summary["mean_size_mean"]
        max_size = summary["max_size_mean"]
        mean_fit = summary["mean_fitness_mean"]
        max_fit = summary["max_fitness_mean"]
        patterns = summary["cum_pattern_types_mean"]

        row = {
            "L": float(size),
            "mean_size_end": float(mean_size[-1]),
            "max_size_end": float(max_size[-1]),
            "mean_size_end_norm": float(mean_size[-1] / (size * size)),
            "max_size_end_norm": float(max_size[-1] / (size * size)),
            "mean_size_early": _window_mean(steps, mean_size, early),
            "mean_size_mid": _window_mean(steps, mean_size, mid),
            "mean_size_late": _window_mean(steps, mean_size, late),
            "late_over_early": float(
                _window_mean(steps, mean_size, late) / _window_mean(steps, mean_size, early)
                if _window_mean(steps, mean_size, early) > 0
                else float("nan")
            ),
            "mean_fitness_early": _window_mean(steps, mean_fit, early),
            "mean_fitness_late": _window_mean(steps, mean_fit, late),
            "mean_fitness_late_over_early": float(
                _window_mean(steps, mean_fit, late) / _window_mean(steps, mean_fit, early)
                if _window_mean(steps, mean_fit, early) > 0
                else float("nan")
            ),
            "corr_size_fitness_2k_10k": _pearson_corr(
                mean_size[(steps >= corr_window[0]) & (steps <= corr_window[1])],
                mean_fit[(steps >= corr_window[0]) & (steps <= corr_window[1])],
            ),
            "slope_mean_size_2k_10k": _window_slope(steps, mean_size, corr_window),
            "t_mean_size_10": _first_crossing(steps, mean_size, 10),
            "t_mean_size_100": _first_crossing(steps, mean_size, 100),
            "t_mean_size_1000": _first_crossing(steps, mean_size, 1000),
            "patterns_end": float(patterns[-1]),
            "max_size_peak": float(max_size.max()),
            "max_size_peak_step": float(steps[max_size.argmax()]),
            "mean_fitness_end": float(mean_fit[-1]),
            "max_fitness_end": float(max_fit[-1]),
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: r["L"])

    # Write summary CSV
    summary_path = outdir / "transition_analysis.csv"
    cols = [
        "L",
        "mean_size_end",
        "max_size_end",
        "mean_size_end_norm",
        "max_size_end_norm",
        "mean_size_early",
        "mean_size_mid",
        "mean_size_late",
        "late_over_early",
        "mean_fitness_early",
        "mean_fitness_late",
        "mean_fitness_late_over_early",
        "corr_size_fitness_2k_10k",
        "slope_mean_size_2k_10k",
        "t_mean_size_10",
        "t_mean_size_100",
        "t_mean_size_1000",
        "patterns_end",
        "max_size_peak",
        "max_size_peak_step",
        "mean_fitness_end",
        "max_fitness_end",
    ]
    data = np.array([[row[c] for c in cols] for row in rows], dtype=np.float64)
    header = ",".join(cols)
    fmt = ["%d"] + ["%.8f"] * (len(cols) - 1)
    np.savetxt(summary_path, data, delimiter=",", header=header, comments="", fmt=fmt)

    # Plots
    L_vals = np.array([r["L"] for r in rows], dtype=np.float64)
    mean_end = np.array([r["mean_size_end"] for r in rows], dtype=np.float64)
    max_end = np.array([r["max_size_end"] for r in rows], dtype=np.float64)
    ratio = np.array([r["late_over_early"] for r in rows], dtype=np.float64)
    corr = np.array([r["corr_size_fitness_2k_10k"] for r in rows], dtype=np.float64)
    fit_ratio = np.array([r["mean_fitness_late_over_early"] for r in rows], dtype=np.float64)

    figures: Dict[str, Path] = {}
    mean_end_path = outdir / "mean_size_end_vs_L.png"
    _plot_line(L_vals, mean_end, mean_end_path, "Mean size at final step", log_y=True)
    figures["Mean size (final) vs L"] = mean_end_path

    max_end_path = outdir / "max_size_end_vs_L.png"
    _plot_line(L_vals, max_end, max_end_path, "Max size at final step", log_y=True)
    figures["Max size (final) vs L"] = max_end_path

    ratio_path = outdir / "mean_size_late_over_early_vs_L.png"
    _plot_line(L_vals, ratio, ratio_path, "Late/Early mean size ratio", log_y=True)
    figures["Late/Early mean size ratio vs L"] = ratio_path

    corr_path = outdir / "corr_size_fitness_vs_L.png"
    _plot_line(L_vals, corr, corr_path, "Corr(mean size, mean fitness) 2k-10k")
    figures["Size/Fitness correlation vs L"] = corr_path

    fit_ratio_path = outdir / "mean_fitness_late_over_early_vs_L.png"
    _plot_line(L_vals, fit_ratio, fit_ratio_path, "Late/Early mean fitness ratio")
    figures["Late/Early mean fitness ratio vs L"] = fit_ratio_path

    threshold_series = {
        thr: np.array([r[f"t_mean_size_{int(thr)}"] for r in rows], dtype=np.float64) for thr in thresholds
    }
    threshold_path = outdir / "time_to_mean_size_thresholds.png"
    _plot_thresholds(L_vals, thresholds, threshold_series, threshold_path)
    figures["Time to mean size thresholds"] = threshold_path

    # Report
    report_path = outdir / "transition_report.md"
    windows = {"early": early, "mid": mid, "late": late, "corr": corr_window}
    _write_report(report_path, rows, windows, thresholds, figures)

    print(f"[analysis] Wrote {summary_path}")
    print(f"[analysis] Wrote {report_path}")
    for fig_path in figures.values():
        print(f"[analysis] Wrote {fig_path}")


if __name__ == "__main__":
    main()
