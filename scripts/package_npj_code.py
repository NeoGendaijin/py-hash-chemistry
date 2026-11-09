#!/usr/bin/env python3
"""Package the public source worktree, excluding gitignored revision material."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files = {ROOT / p for p in tracked if p and (ROOT / p).is_file()}
    for pattern in ("scripts/*.py", "scripts/*.sh", "src/*.py", "src/*.c"):
        files.update(ROOT.glob(pattern))
    names = sorted(p.relative_to(ROOT).as_posix() for p in files)
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "-z", "--stdin"], cwd=ROOT,
        input=("\0".join(names) + "\0").encode(), stdout=subprocess.PIPE, check=False,
    )
    if ignored.returncode not in (0, 1):
        ignored.check_returncode()
    excluded = set(ignored.stdout.decode().split("\0"))
    files = sorted(ROOT / name for name in names if name not in excluded)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    metadata = dict(code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    tracked_worktree_dirty=bool(subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT)),
                    source_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in files},
                    note="Exact public-source worktree snapshot; gitignored material is excluded. "
                         "Source hashes, not the HEAD alone, identify this code.")
    target = ROOT / "dist/npj-complexity-code.tar.gz"
    with target.open("wb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=1780000000) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for path in files:
                info = archive.gettarinfo(str(path), "py-hash-chemistry/" + path.relative_to(ROOT).as_posix())
                info.mtime, info.uid, info.gid, info.uname, info.gname = 1780000000, 0, 0, "", ""
                with path.open("rb") as data:
                    archive.addfile(info, data)
            data = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
            info = tarfile.TarInfo("py-hash-chemistry/CODE_PROVENANCE.json")
            info.size, info.mtime, info.mode = len(data), 1780000000, 0o644
            archive.addfile(info, io.BytesIO(data))
    checksum = digest(target)
    target.with_name(target.name + ".sha256").write_text(f"{checksum}  {target.name}\n")
    print(f"{len(files)} source files: {checksum}  {target}")


if __name__ == "__main__":
    main()
