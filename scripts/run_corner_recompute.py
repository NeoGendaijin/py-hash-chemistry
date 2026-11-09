#!/usr/bin/env python3
"""Run the remaining corner-corrected revision datasets on GPUs 0, 1 and 3.

GPU 2 is reserved for the desktop. All data and logs are separate from the
historical datasets; failures and the thermal stop file stop subsequent jobs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/corner_fixed"
PY = ["uv", "run", "--no-sync", "python"]


def worker(gpu, jobs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cuda",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    for name, script, args in jobs:
        if (OUT / "THERMAL_STOP").exists():
            raise RuntimeError("thermal stop file exists")
        command = [*PY, "-u", "scripts/" + script, *args]
        print(f"GPU {gpu} START {name}", flush=True)
        start = time.time()
        with (OUT / f"{name}.log").open("a", buffering=1) as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt = dict(name=name, gpu=gpu, command=command, returncode=result.returncode,
                       elapsed_seconds=time.time() - start)
        (OUT / f"{name}.completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {OUT / (name + '.log')}")
        print(f"GPU {gpu} DONE {name} ({receipt['elapsed_seconds']:.0f}s)", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {g: [(f"sweep_part_{i}", "run_spatial_dyadic_sweep.py",
                 ["--part", f"{i}/3", "--output-root", "results/corner_fixed/spatial_dyadic"])]
            for i, g in enumerate((0, 1, 3))}
    for gpu, arm in ((0, "legacy"), (1, "fixed"), (3, "fixed_mix")):
        jobs[gpu].append((f"original_{arm}", "compare_original_schc.py",
                          ["--arm", arm, "--seeds", "0-99", "--batch", "50",
                           "--output-root", "results/corner_fixed/original_comparison"]))
    jobs[0].append(("snapshots", "run_dyadic_snapshots.py",
                    ["--output-root", "results/corner_fixed/spatial_dyadic_snapshots",
                     "--sweep-root", "results/corner_fixed/spatial_dyadic"]))
    jobs[1].append(("validation_L100", "run_transition_batched.py",
                    ["--size", "100", "--seeds", "0-49", "--steps", "20000",
                     "--stop-mean-size", "0", "--output-root", "results/corner_fixed/transition_fixed_check"]))
    guard = subprocess.Popen([*PY, "scripts/thermal_watchdog.py", "--pattern", "results/corner_fixed",
                              "--log", str(OUT / "thermal.csv"), "--stop-file", str(OUT / "THERMAL_STOP"),
                              "--grace", "120"], cwd=ROOT)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(worker, gpu, tasks) for gpu, tasks in jobs.items()]
            for future in futures:
                future.result()
    finally:
        guard.terminate()
        guard.wait()
    print("All corner-corrected datasets finished", flush=True)
    subprocess.run([*PY, "-u", "scripts/finalize_corner_recompute.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
