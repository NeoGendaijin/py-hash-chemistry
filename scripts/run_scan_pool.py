#!/usr/bin/env python3
"""Run a (size, seed) scan via fast_scan.py across multiple GPUs.

Maintains one worker per GPU; each worker pulls (L, seed) jobs from a shared
queue and runs fast_scan.py pinned to its GPU. Reusable for open or periodic
boundary scans, sensitivity sweeps, etc.

Example:
  python scripts/run_scan_pool.py --periodic \
      --sizes 200 240 280 300 320 360 400 --seeds 0-9 --steps 20000 \
      --output-root results/boundary_control/periodic --gpus 0 1 2 3
"""
from __future__ import annotations
import argparse, os, queue, subprocess, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_seeds(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=str, default="0-9", help="e.g. 0-9 or 0,1,2")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--sample-interval", type=int, default=10)
    ap.add_argument("--mu", type=str, default="0.002/0.999")
    ap.add_argument("--death-prob", type=float, default=0.001)
    ap.add_argument("--periodic", action="store_true")
    ap.add_argument("--stop-mean-size", type=float, default=0.0,
                    help="Early-stop a run once mean component size exceeds this (0=off)")
    ap.add_argument("--mem-fraction", type=float, default=0.5,
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION per worker (cap GPU mem preallocation)")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1],
                    help="GPU ids to use (default 2 GPUs, conservative for power/thermal)")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    jobs = queue.Queue()
    for L in args.sizes:
        for s in seeds:
            jobs.put((L, s))
    total = jobs.qsize()
    print(f"[pool] {total} jobs over GPUs {args.gpus} "
          f"(periodic={args.periodic}, steps={args.steps})", flush=True)

    done = {"n": 0}
    lock = threading.Lock()

    def worker(gpu_id):
        while True:
            try:
                L, seed = jobs.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id),
                       XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.mem_fraction))
            cmd = [sys.executable, str(ROOT / "scripts" / "fast_scan.py"),
                   "--size", str(L), "--seed", str(seed), "--steps", str(args.steps),
                   "--sample-interval", str(args.sample_interval),
                   "--mu", args.mu, "--death-prob", str(args.death_prob),
                   "--output-root", str(args.output_root),
                   "--stop-mean-size", str(args.stop_mean_size)]
            if args.periodic:
                cmd.append("--periodic")
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            with lock:
                done["n"] += 1
                tag = "ok" if r.returncode == 0 else f"FAIL({r.returncode})"
                print(f"[{done['n']}/{total}] gpu{gpu_id} L={L} seed={seed} {tag}", flush=True)
                if r.returncode != 0:
                    print(r.stderr[-500:], flush=True)
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("[pool] all jobs complete", flush=True)


if __name__ == "__main__":
    main()
