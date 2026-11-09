#!/usr/bin/env python3
"""Preserve the initial attempt and retain only fully completed conditions for resume."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/corner_fixed"
BACKUP = ROOT / "results/corner_fixed_initial_batch"


def main():
    shutil.copytree(BASE, BACKUP)
    complete = []
    for directory in sorted((BASE / "spatial_dyadic").glob("*/*")):
        if not directory.is_dir():
            continue
        paths = [directory / f"seed_{s}.csv" for s in range(30)]
        ok = all(p.exists() for p in paths)
        if ok:
            for path in paths:
                with path.open() as stream:
                    ok &= sum(1 for _ in stream) == 2001
        if ok:
            complete.append(directory.relative_to(BASE).as_posix())
        else:
            target = BACKUP / "aborted" / directory.relative_to(BASE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(directory), str(target))
    print(f"Preserved {len(complete)} complete conditions ({30 * len(complete)} runs)")
    print("\n".join(complete))


if __name__ == "__main__":
    main()
