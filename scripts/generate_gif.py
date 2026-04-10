#!/usr/bin/env python3
"""Generate detailed GIF animations of SCHC evolution.

Saves frames every N steps and assembles into GIF with largest
connected component highlighted in black.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from jax import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import SCHCParams, advance_one_step_jit, initialize_state


def find_largest_component(config_np: np.ndarray) -> np.ndarray:
    occupied = config_np > 0
    if not occupied.any():
        return np.zeros_like(occupied)
    labeled, n = ndimage.label(occupied, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(occupied)
    sizes = ndimage.sum(occupied, labeled, range(1, n + 1))
    return labeled == (np.argmax(sizes) + 1)


def config_to_image(config_np: np.ndarray) -> np.ndarray:
    """Convert grid state to RGB image with largest component in black."""
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
    return img


PAPER_STEPS = [2000, 5000, 10000]  # steps to save config arrays for fig2


def run_and_gif(size: int, seed: int, max_steps: int, interval: int,
                k: int, n: int, mu: float, death_prob: float,
                outdir: Path):
    """Run simulation and save GIF + config arrays for paper figures."""
    params = SCHCParams(k=k, n=n, L=size, mu=mu, death_prob=death_prob)
    key = random.PRNGKey(seed)
    state = initialize_state(key, params)

    frames = []
    frame_dir = outdir / f"frames_L{size}_seed{seed}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    config_dir = outdir / f"configs_L{size}_seed{seed}"
    config_dir.mkdir(parents=True, exist_ok=True)

    for t in range(max_steps + 1):
        if t % interval == 0:
            config_np = np.array(state.config)
            img = config_to_image(config_np)
            img_uint8 = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            frames.append(pil_img)
            pil_img.save(frame_dir / f"frame_{t:05d}.png")
            largest_size = int(find_largest_component(config_np).sum())
            print(f"[frame] L={size} step={t} max_comp={largest_size}", flush=True)

        if t in PAPER_STEPS:
            config_np = np.array(state.config)
            np.save(config_dir / f"config_step{t:05d}.npy", config_np)
            print(f"[config saved] L={size} step={t}", flush=True)

        if t == max_steps:
            break
        state = advance_one_step_jit(state, params)

    # Assemble GIF
    gif_path = outdir / f"schc_L{size}_seed{seed}.gif"
    durations = [100] * (len(frames) - 1) + [2000]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)
    print(f"[done] Saved {gif_path} ({len(frames)} frames)")
    return gif_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 8])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--mu", type=str, default="0.002/0.999")
    parser.add_argument("--death-prob", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "gifs")
    args = parser.parse_args()
    mu = eval(args.mu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        for seed in args.seeds:
            print(f"\n=== L={size}, seed={seed} ===")
            run_and_gif(size, seed, args.steps, args.interval,
                        args.k, args.n, mu, args.death_prob, args.output_dir)


if __name__ == "__main__":
    main()
