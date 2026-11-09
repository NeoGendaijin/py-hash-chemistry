#!/usr/bin/env python3
"""Package corrected revision data with an inventory and reproducible checksums.

Excludes superseded transition/periodic data, shared email attachments, compiled
libraries, correspondence and working Markdown reports. Main figures need no
private inputs; the public data README is included explicitly.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
NAME = "npj-complexity-data.tar.gz"
FIXED_MTIME = 1_780_000_000
SIZES = [100, 120, 140, 160, 180, 200, 210, 220, 230, 240, 250,
         260, 280, 300, 320, 340, 360, 380, 400]
JOBS = ("sweep_part_0", "sweep_part_1", "sweep_part_2", "original_legacy", "original_fixed",
        "original_fixed_mix", "snapshots", "validation_L100")
WITHDRAWN_OUTPUTS = {"npj_rev/ext1/contrasts_original_vs_best.csv"}
TREES = {
    "cpu_scan_full/original_mix": ("*.csv",),
    "cpu_scan/legacy_xor": ("*.csv",),
    "cpu_scan/original_mix": ("*.csv",),
    "cpu_novelty": ("*.csv",),
    "cpu_sensitivity": ("*.csv",),
    "cpu_scan_40k": ("*.csv",),
    "cpu_snapshots": ("*.csv", "*.npy"),
    "spatial_dyadic/intended": ("*.csv",),
    "spatial_dyadic/as_coded": ("*.csv",),
    "spatial_dyadic/stats": ("*.csv",),
    "spatial_dyadic_snapshots/intended": ("*.npy",),
    "original_comparison": ("*.csv",),
    "corner_fixed/spatial_dyadic": ("*.csv", "*.json"),
    "corner_fixed/spatial_dyadic_snapshots": ("*.npy", "*.json"),
    "corner_fixed/original_comparison": ("*.csv", "*.json"),
    "corner_fixed/transition_fixed_check": ("*.csv", "*.json"),
    "corner_fixed/source_snapshot": ("*.py",),
    "transition_fixed": ("*.csv",),
    "transition_fixed_check": ("*.csv",),
    "cpu_scan_check": ("*.csv",),
    "cpu_scan_L100": ("*.csv",),
    "npj_rev": ("*.csv", "*.tex", "*.json"),
    "large_space/L200": ("*.csv",),
    "large_space/L400": ("*.csv",),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_required_runs():
    for job in JOBS:
        record = RES / f"corner_fixed/{job}.completion.json"
        if not record.exists() or json.loads(record.read_text())["returncode"] != 0:
            raise ValueError(f"unfinished corrected calculation: {job}")
    missing = []
    for size in SIZES:
        for seed in range(100):
            p = RES / f"cpu_scan_full/original_mix/L{size}/runs/seed_{seed}.csv"
            if not p.exists():
                missing.append(p)
    for rule in ("intended", "as_coded"):
        for d in range(10, 101, 10):
            for pi in range(6):
                if rule == "as_coded" and pi == 0:
                    continue
                for seed in range(30):
                    p = RES / f"corner_fixed/spatial_dyadic/{rule}/D{d}_P{pi / 5:.1f}/seed_{seed}.csv"
                    if not p.exists():
                        missing.append(p)
    if missing:
        raise FileNotFoundError(f"{len(missing)} required runs missing; first: {missing[0]}")


def collect() -> list[Path]:
    check_required_runs()
    files = set()
    for dirname, patterns in TREES.items():
        directory = RES / dirname
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        found = {p for pat in patterns for p in directory.rglob(pat)
                 if ".partial." not in p.name and p.relative_to(RES).as_posix() not in WITHDRAWN_OUTPUTS}
        if not found:
            raise FileNotFoundError(f"no matching data under {directory}")
        files.update(found)
    files.update(RES / f"corner_fixed/{job}.completion.json" for job in JOBS)
    files.update(RES / f"corner_fixed/{name}" for name in ("thermal.csv", "adaptive_validation.json"))
    assert all(p.is_file() for p in files)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def provenance(files: list[Path]) -> bytes:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    packages = {}
    for name in ("jax", "jaxlib", "numpy", "pandas", "scipy", "matplotlib", "seaborn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    sources = sorted(set(ROOT.glob("src/*.py")) | set(ROOT.glob("src/*.c"))
                     | set(ROOT.glob("scripts/*.py")) | set(ROOT.glob("scripts/*.sh")))
    inventory = {}
    for f in files:
        rel = f.relative_to(RES)
        group = "/".join(rel.parts[:2])
        inventory[group] = inventory.get(group, 0) + 1
    data = {
        "schema": 2,
        "release": "npj-complexity-rev1",
        "code_commit": git("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(git("diff", "HEAD", "--name-only")),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
        "python": platform.python_version(),
        "analysis_packages": packages,
        "data_files": len(files),
        "inventory": inventory,
        "extension_I": {
            "engine": "JAX", "L": 100, "seeds": "0-29", "steps": 2000,
            "D": list(range(10, 101, 10)), "P": [i / 5 for i in range(6)],
            "winner_rule": "original", "score": "mix", "legacy_corner": False,
            "directory": "corner_fixed/spatial_dyadic",
            "row_time": "row t records events of step t+1; n_active is measured before that step",
        },
        "extension_II": {
            "engine": "C", "sizes": SIZES, "seeds": "0-99", "steps": 20000,
            "sample_interval": 10, "stop_mean_size": 0,
            "winner_rule": "original", "score": "mix", "periodic": False,
        },
        "novelty": {"L": 200, "seeds": "0-39", "steps": 20000, "sample_interval": 1},
        "sensitivity": {"sizes": [100, 200, 240, 300, 320, 400], "seeds": "0-29", "steps": 10000},
        "longer_horizon": {"sizes": [240, 320, 400], "seeds": "0-99", "steps": 40000,
                           "stop_mean_size": 2000},
        "reference_data": {
            "original_schc": "https://github.com/hsayama/Structural-Cellular-Hash-Chemistry",
            "spatial_dyadic": "https://github.com/hsayama/Spatial-Dyadic-Structural-Cellular-Hash-Chemistry",
            "private_sd_csv": "Not redistributed; available from H. Sayama on reasonable request",
        },
        "caveat": "Source hashes identify the reproduction code at packaging, not a retrospectively invented experiment commit.",
    }
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()


def add_bytes(tar, arcname, data):
    info = tarfile.TarInfo(arcname)
    info.size, info.mtime, info.mode = len(data), FIXED_MTIME, 0o644
    tar.addfile(info, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "dist")
    args = ap.parse_args()
    files = collect()
    entries = {
        "results/README_npj_data.md": (ROOT / "docs/npj_data_README.md").read_bytes(),
        "results/PROVENANCE_npj.json": provenance(files),
    }
    entries["results/MANIFEST_npj.sha256"] = ("".join(
        f"{hashlib.sha256(value).hexdigest()}  {key}\n" for key, value in sorted(entries.items()))
        + "".join(f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in files)).encode()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / NAME
    with out.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb",
                                             mtime=FIXED_MTIME, compresslevel=6) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, content in sorted(entries.items()):
                add_bytes(tar, name, content)
            for p in files:
                info = tar.gettarinfo(str(p), arcname=p.relative_to(ROOT).as_posix())
                info.mtime, info.uid, info.gid, info.uname, info.gname = FIXED_MTIME, 0, 0, "", ""
                info.mode = 0o644
                with p.open("rb") as fh:
                    tar.addfile(info, fh)
    digest = sha256(out)
    (args.out_dir / f"{NAME}.sha256").write_text(f"{digest}  {NAME}\n")
    (args.out_dir / "PROVENANCE_npj.json").write_bytes(entries["results/PROVENANCE_npj.json"])
    raw_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"{len(files)} data files ({raw_mb:.0f} MB raw) -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
