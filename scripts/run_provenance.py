"""Freeze the model and source identity alongside a simulation dataset."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import jax

ROOT = Path(__file__).resolve().parents[1]
FIXED_ROOT = ROOT / "results" / "corner_fixed"


def archived_source_matches(name, digest):
    candidates = [ROOT / name, *(FIXED_ROOT / "source_snapshot").rglob(Path(name).name)]
    return any(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest for p in candidates)


def complete_outputs(directory, definition):
    seeds, steps = definition["seeds"], definition["steps"]
    if "save_steps" in definition:
        return all((directory / f"seed_{seed}_t{t}.npy").is_file()
                   for seed in seeds for t in definition["save_steps"])
    for seed in seeds:
        path = directory / f"seed_{seed}.csv"
        if not path.is_file():
            return False
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            return False
        times = [int(row["step"]) for row in rows]
        if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
            return False
        if "sample_interval" not in definition:
            if len(rows) != steps or times[-1] != steps - 1:
                return False
        else:
            last = rows[-1]
            threshold = definition.get("stop_mean_size", 0)
            if not (times[-1] == steps or int(last["n_active"]) == 0 or
                    (threshold > 0 and float(last["mean_size"]) >= threshold)):
                return False
    return True


def register_run(directory, params, seeds, steps, **extra):
    directory = Path(directory)
    definition = dict(model=asdict(params), seeds=list(seeds), steps=steps, **extra)
    sources = [ROOT / "src/simulation.py", ROOT / "src/metrics.py",
               ROOT / "scripts/compare_original_schc.py", ROOT / "scripts/adaptive_batch.py",
               ROOT / "scripts/run_spatial_dyadic_sweep.py", ROOT / "scripts/run_dyadic_snapshots.py",
               ROOT / "scripts/run_transition_batched.py", Path(__file__)]
    identity = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources}
    path = directory / "run_metadata.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["definition"] != definition:
            raise ValueError(f"incompatible existing simulation metadata: {path}")
        if old["source_sha256"] != identity:
            # Completed pre-optimization runs retain their actual source identity.
            # Never attach the new implementation's hash to old trajectories.
            for name, digest in old["source_sha256"].items():
                if not archived_source_matches(name, digest):
                    raise ValueError(f"unavailable archived simulation source: {name}")
            if not complete_outputs(directory, definition):
                raise ValueError(f"unfinished run requires its original implementation: {path}")
        return
    if directory.exists() and any(directory.glob("seed_*")):
        raise ValueError(f"refusing to mix unversioned data with a new run: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(definition=definition, source_sha256=identity,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    jax_version=jax.__version__, devices=list(map(str, jax.devices())))
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
        name = stream.name
    os.replace(name, path)
