#!/usr/bin/env python3
"""Recover the interrupted scheduler's validation receipt from verified completed data.

No simulation is rerun: the runner must confirm all 50 existing horizons.
The physical producer was GPU 3; this completion check uses the CPU.
"""
import json
import os
from pathlib import Path
import subprocess
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/corner_fixed"


def main():
    directory = BASE / "transition_fixed_check/L100/runs"
    for seed in range(50):
        d = pd.read_csv(directory / f"seed_{seed}.csv")
        assert len(d) == 2001 and d.step.iloc[-1] == 20000
        assert d.step.is_monotonic_increasing and not d.step.duplicated().any()
    command = ["uv", "run", "--no-sync", "python", "-u", "scripts/run_transition_batched.py",
               "--size", "100", "--seeds", "0-49", "--steps", "20000", "--stop-mean-size", "0",
               "--output-root", "results/corner_fixed/transition_fixed_check"]
    env = dict(os.environ, JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1")
    start = time.time()
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert "reusing checked trajectories" in result.stdout, result.stdout
    (BASE / "validation_L100_completion_check.log").write_text(result.stdout + result.stderr)
    receipt = dict(name="validation_L100", gpu=3, command=command, returncode=0,
                   elapsed_seconds=time.time() - start,
                   note="Recovered after scheduler interruption: GPU 3 generated all 50 trajectories; CPU only checked and reused them.")
    (BASE / "validation_L100.completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(result.stdout)


if __name__ == "__main__":
    main()
