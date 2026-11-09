#!/usr/bin/env python3
"""Regenerate numerical results after all corrected simulation jobs succeed.

This deliberately does not edit manuscript prose or claim the revision is done.
The resulting summaries must be checked and incorporated before building PDFs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/corner_fixed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="wait for all eight successful completion receipts")
    args = parser.parse_args()
    names = ("sweep_part_0", "sweep_part_1", "sweep_part_2", "original_legacy", "original_fixed",
             "original_fixed_mix", "snapshots", "validation_L100")
    if args.wait:
        previous = None
        while True:
            ready = [name for name in names if (BASE / f"{name}.completion.json").exists()
                     and json.loads((BASE / f"{name}.completion.json").read_text())["returncode"] == 0]
            runs = len(list((BASE / "spatial_dyadic").rglob("seed_*.csv")))
            status = (len(ready), runs)
            if status != previous:
                print(f"Waiting: {len(ready)}/8 jobs complete; {runs}/3300 sweep runs saved", flush=True)
                previous = status
            if len(ready) == len(names):
                break
            time.sleep(30)
    for name in names:
        receipt = json.loads((BASE / f"{name}.completion.json").read_text())
        if receipt["returncode"]:
            raise RuntimeError(f"simulation job failed: {name}")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLCONFIGDIR="/tmp/npj-mpl")
    for name in ("analyze_extension1_stats", "npj_rev_ext1", "analyze_original_comparison",
                 "validate_cpu_engine", "npj_revision_supplement", "analyze_corner_recompute"):
        if name == "analyze_extension1_stats" and not (ROOT / "tex/Large-Hash-Chemistry/review/Sayama-addition/SD-SCHC-experimental-data.csv").exists():
            print("Private Mathematica sweep CSV absent: skipping its optional comparison.", flush=True)
            continue
        if name == "analyze_original_comparison" and not (ROOT.parent / "Structural-Cellular-Hash-Chemistry/maxFitnessSeries.dat").exists():
            print("Public Mathematica series checkout absent: skipping its optional comparison.", flush=True)
            continue
        if name == "analyze_corner_recompute" and not (ROOT / "results/npj_rev/corner_check/previous_per_run.csv").exists():
            print("Historical sweep summary absent: skipping its optional paired-change audit.", flush=True)
            continue
        command = ["uv", "run", "--no-sync", "python", f"scripts/{name}.py"]
        with (BASE / f"{name}.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f"Updated {name}", flush=True)
    print("Numerical results ready; manuscript prose and final build still require review.", flush=True)


if __name__ == "__main__":
    main()
