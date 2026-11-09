#!/usr/bin/env python3
"""Use GPU 3 for validation after its own sweep and reference comparison finish."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/corner_fixed"


def main():
    while True:
        receipt = BASE / "original_fixed_mix.completion.json"
        if receipt.exists() and json.loads(receipt.read_text())["returncode"] == 0:
            break
        if (BASE / "THERMAL_STOP").exists():
            raise RuntimeError("thermal stop file exists")
        time.sleep(10)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="3", JAX_PLATFORMS="cuda", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    command = ["uv", "run", "--no-sync", "python", "-u", "scripts/run_transition_batched.py",
               "--size", "100", "--seeds", "0-49", "--steps", "20000", "--stop-mean-size", "0",
               "--output-root", "results/corner_fixed/transition_fixed_check"]
    print("Starting the scheduled L100 validation early on GPU 3", flush=True)
    with (BASE / "validation_L100_early.log").open("w") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print("Early validation complete; the regular scheduler will verify and reuse it.", flush=True)


if __name__ == "__main__":
    main()
