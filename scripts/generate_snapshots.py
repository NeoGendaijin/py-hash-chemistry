#!/usr/bin/env python3
"""
Generate grid state snapshots as PNG images for SCHC at L=200 and L=400.
Saves snapshots at steps 0, 100, 500, 2000, 5000, 10000, 20000.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from jax import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import SCHCParams, advance_one_step_jit, initialize_state


SNAPSHOT_STEPS = [0, 100, 500, 2000, 5000, 10000, 20000]


def find_largest_component(config_np: np.ndarray) -> np.ndarray:
    """Find the largest connected component (8-connectivity) using scipy.
    Returns a boolean mask of the largest component."""
    from scipy import ndimage
    occupied = config_np > 0
    if not occupied.any():
        return np.zeros_like(occupied)
    labeled, n_components = ndimage.label(occupied, structure=np.ones((3, 3)))
    if n_components == 0:
        return np.zeros_like(occupied)
    sizes = ndimage.sum(occupied, labeled, range(1, n_components + 1))
    largest_label = np.argmax(sizes) + 1
    return labeled == largest_label


def largest_component_outline(config_np: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the boundary pixels of the largest component.
    A pixel is on the boundary if it belongs to the largest component and
    at least one of its 8-neighbors does not."""
    from scipy.ndimage import binary_erosion
    largest = find_largest_component(config_np)
    if not largest.any():
        return largest
    interior = binary_erosion(largest, structure=np.ones((3, 3)))
    return largest & ~interior


def make_colormap(k: int):
    """Create a colormap: 0=white (empty), 1..k=HSV colors."""
    colors = [np.array([1.0, 1.0, 1.0, 1.0])]  # white for empty
    np.random.seed(42)
    for i in range(k):
        hue = (i * 0.618033988749895) % 1.0  # golden ratio for spread
        sat = 0.7 + 0.3 * np.random.random()
        val = 0.6 + 0.4 * np.random.random()
        # HSV to RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append(np.array([r, g, b, 1.0]))
    return colors


def save_snapshot(config_np: np.ndarray, outpath: Path, step: int, size: int, k: int):
    """Save a single grid snapshot as PNG."""
    L = config_np.shape[0]
    # Create RGB image
    img = np.ones((L, L, 3), dtype=np.float32)  # white background

    occupied = config_np > 0
    if occupied.any():
        # Map cell types to colors using golden-ratio hue
        types = config_np[occupied].astype(np.float64)
        hues = (types * 0.618033988749895) % 1.0
        sats = np.full_like(hues, 0.8)
        vals = np.full_like(hues, 0.85)
        # HSV to RGB vectorized
        hsv = np.stack([hues, sats, vals], axis=-1)
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(hsv)
        img[occupied] = rgb

        # Outline the largest connected component in black
        outline = largest_component_outline(config_np)
        img[outline] = [0.0, 0.0, 0.0]  # black outline

    largest_size = int(find_largest_component(config_np).sum()) if occupied.any() else 0
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img, interpolation="nearest", origin="upper")
    ax.set_title(f"L={size}, step={step}, max comp.={largest_size}", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def save_composite(snapshots: dict, outdir: Path, size: int, k: int):
    """Save a composite figure with all snapshots side by side."""
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, (step, config_np) in enumerate(sorted(snapshots.items())):
        ax = axes[idx]
        L = config_np.shape[0]
        img = np.ones((L, L, 3), dtype=np.float32)
        occupied = config_np > 0
        if occupied.any():
            types = config_np[occupied].astype(np.float64)
            hues = (types * 0.618033988749895) % 1.0
            sats = np.full_like(hues, 0.8)
            vals = np.full_like(hues, 0.85)
            hsv = np.stack([hues, sats, vals], axis=-1)
            from matplotlib.colors import hsv_to_rgb
            rgb = hsv_to_rgb(hsv)
            img[occupied] = rgb
            outline = largest_component_outline(config_np)
            img[outline] = [0.0, 0.0, 0.0]
        ax.imshow(img, interpolation="nearest", origin="upper")
        ax.set_title(f"t={step}", fontsize=12)
        ax.axis("off")

    fig.suptitle(f"SCHC Grid Snapshots (L={size})", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / f"composite_L{size}.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_and_snapshot(size: int, seed: int, steps: int, k: int, n: int,
                     mu: float, death_prob: float, outdir: Path,
                     progress_interval: int = 500):
    """Run simulation and save snapshots at specified steps."""
    params = SCHCParams(k=k, n=n, L=size, mu=mu, death_prob=death_prob)
    key = random.PRNGKey(seed)
    state = initialize_state(key, params)

    size_dir = outdir / f"L{size}"
    size_dir.mkdir(parents=True, exist_ok=True)

    snapshots = {}
    max_step = max(SNAPSHOT_STEPS)
    actual_steps = min(steps, max_step)

    for t in range(actual_steps + 1):
        if t in SNAPSHOT_STEPS:
            config_np = np.array(state.config)
            snapshots[t] = config_np.copy()
            save_snapshot(config_np, size_dir / f"step_{t:05d}.png", t, size, k)
            print(f"[snapshot] L={size} step={t} saved", flush=True)

        if progress_interval and t % progress_interval == 0 and t not in SNAPSHOT_STEPS:
            print(f"[progress] L={size} step={t}", flush=True)

        if t == actual_steps:
            break
        state = advance_one_step_jit(state, params)

    save_composite(snapshots, size_dir, size, k)
    print(f"[done] L={size}: saved {len(snapshots)} snapshots to {size_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate grid state snapshots for SCHC.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--mu", type=str, default="0.002/0.999")
    parser.add_argument("--death-prob", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "snapshots")
    parser.add_argument("--progress-interval", type=int, default=500)
    args = parser.parse_args()

    mu_val = float(eval(str(args.mu), {"__builtins__": None}, {}))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        run_and_snapshot(size, args.seed, args.steps, args.k, args.n,
                         mu_val, args.death_prob, args.output_dir,
                         args.progress_interval)


if __name__ == "__main__":
    main()
