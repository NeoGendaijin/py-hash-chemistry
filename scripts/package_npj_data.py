#!/usr/bin/env python3
"""Build the data archive attached to the npj Complexity code release.

Collects the processed outputs behind the figures and tables of the SCHC
finite-size study, adds docs/npj_data_README.md and a SHA-256 manifest, and
writes a reproducible tar.gz (sorted members, fixed mtime/owner) to dist/.

    python scripts/package_npj_data.py
    tar -tzf dist/npj-complexity-data.tar.gz | head

Excluded on purpose: results/boundary_control/periodic (superseded run with a
component-detection bug; the paper uses periodic_v2) and results/alt_hash
(not used in the paper).
"""
from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DIST = ROOT / "dist"
NAME = "npj-complexity-data.tar.gz"

CSV_DIRS = [
    "transition_scan",
    "fine_transition_scan",
    "boundary_control/periodic_v2",
    "large_space",
    "param_sensitivity_mu",
    "param_sensitivity_death",
    "statistical_analysis",
]
SNAPSHOT_DIRS = ["gifs/configs_L200_seed0", "gifs/configs_L400_seed8"]
FIXED_MTIME = 1_780_000_000  # constant timestamp so the archive is byte-reproducible


def collect() -> list[Path]:
    files: list[Path] = []
    for d in CSV_DIRS:
        found = sorted((RES / d).rglob("*.csv"))
        if not found:
            raise FileNotFoundError(f"no CSV files under results/{d}")
        files += found
    for d in SNAPSHOT_DIRS:
        found = sorted((RES / d).glob("*.npy"))
        if not found:
            raise FileNotFoundError(f"no snapshots under results/{d}")
        files += found
    return files


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size, info.mtime, info.mode = len(data), FIXED_MTIME, 0o644
    tar.addfile(info, io.BytesIO(data))


def main() -> None:
    files = collect()
    rel = [f.relative_to(ROOT).as_posix() for f in files]
    manifest = "".join(f"{sha256(f)}  {r}\n" for f, r in zip(files, rel))
    readme = (ROOT / "docs" / "npj_data_README.md").read_bytes()

    DIST.mkdir(exist_ok=True)
    out = DIST / NAME
    with open(out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                               mtime=FIXED_MTIME, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            add_bytes(tar, "results/README_npj_data.md", readme)
            add_bytes(tar, "results/MANIFEST_npj.sha256", manifest.encode())
            for f, r in zip(files, rel):
                info = tar.gettarinfo(str(f), arcname=r)
                info.mtime, info.uid, info.gid, info.uname, info.gname = FIXED_MTIME, 0, 0, "", ""
                info.mode = 0o644
                with open(f, "rb") as fh:
                    tar.addfile(info, fh)
    digest = sha256(out)
    (DIST / f"{NAME}.sha256").write_text(f"{digest}  {NAME}\n")
    raw_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"{len(files)} files ({raw_mb:.0f} MB raw) -> {out.relative_to(ROOT)} "
          f"({out.stat().st_size / 1e6:.0f} MB)\nsha256 {digest}")


if __name__ == "__main__":
    main()
