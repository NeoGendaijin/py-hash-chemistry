#!/usr/bin/env python3
"""Check public archives and Git indexes without publishing or deleting files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CODE_MARKDOWN = {"README.md", "docs/npj_data_README.md"}
PRIVATE_PARTS = {"review", "report", "internal", "private", "Sayama-addition"}


def safe_name(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and ".." not in path.parts, name
    assert not PRIVATE_PARTS.intersection(path.parts), name


def digest_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def validate_git():
    markdown = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.md"], cwd=ROOT,
    ).decode().strip("\0").split("\0")
    markdown = {name for name in markdown if name}
    assert markdown <= PUBLIC_CODE_MARKDOWN, sorted(markdown - PUBLIC_CODE_MARKDOWN)
    fixtures = ["report/private-check.md", "review/private-check.md",
                "docs/alife2026_erratum.md", "docs/release_notes_npj.md",
                "results/analysis/private-check.md", "docs/private-check.internal.md"]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-z", "--stdin"], cwd=ROOT,
        input=("\0".join(fixtures) + "\0").encode(), stdout=subprocess.PIPE, check=True,
    )
    assert set(result.stdout.decode().strip("\0").split("\0")) == set(fixtures)
    paper = ROOT / "tex/Large-Hash-Chemistry"
    if (paper / ".git").exists():
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "review/", "*.md"], cwd=paper,
        ).decode().strip("\0")
        assert not tracked, tracked
        for name in ("review/response-to-reviewers.md", "review/cover-letter.md",
                     "review/response-table.md", "review/reviewer1.md", "review/reviewer2.md"):
            assert (paper / name).is_file(), f"local file unexpectedly missing: {name}"
    for name in ("docs/alife2026_erratum.md", "docs/release_notes_npj.md"):
        assert (ROOT / name).is_file(), f"local draft unexpectedly missing: {name}"
    return {"public_tracked_markdown": sorted(markdown), "ignore_fixtures": len(fixtures)}


def validate_code(path):
    hashes, provenance, markdown = {}, None, []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            prefix = "py-hash-chemistry/"
            assert member.name.startswith(prefix), member.name
            name = member.name[len(prefix):]
            safe_name(name)
            if name.endswith(".md"):
                assert name in PUBLIC_CODE_MARKDOWN, name
                markdown.append(name)
            with archive.extractfile(member) as data:
                if name == "CODE_PROVENANCE.json":
                    provenance = json.load(data)
                else:
                    hashes[name] = digest_stream(data)
    assert provenance is not None
    assert hashes == provenance["source_sha256"], "code provenance differs from archive contents"
    return {"source_files": len(hashes), "markdown": sorted(markdown)}


def validate_data(path):
    hashes, expected, markdown = {}, None, []
    manifest_name = "results/MANIFEST_npj.sha256"
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            safe_name(member.name)
            assert member.name != "results/npj_rev/ext1/contrasts_original_vs_best.csv", member.name
            if member.name.endswith(".md"):
                assert member.name == "results/README_npj_data.md", member.name
                markdown.append(member.name)
            with archive.extractfile(member) as data:
                if member.name == manifest_name:
                    expected = dict(line.split("  ", 1)[::-1]
                                    for line in data.read().decode().splitlines())
                else:
                    hashes[member.name] = digest_stream(data)
    assert expected is not None
    assert hashes == expected, "data manifest differs from archive contents"
    return {"manifest_verified_files": len(hashes), "markdown": sorted(markdown)}


def validate_source(path):
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            safe_name(name)
            assert not name.endswith(".md"), name
            assert "P-D-overall-performance" not in name, name
        assert archive.testzip() is None
        return {"files": len(archive.namelist()), "private_material": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-git", action="store_true", help="Only validate supplied archives.")
    parser.add_argument("--code", type=Path, default=ROOT / "dist/npj-complexity-code.tar.gz")
    parser.add_argument("--data", type=Path, default=ROOT / "dist/npj-complexity-data.tar.gz")
    parser.add_argument("--source", type=Path, default=ROOT / "dist/npj-submission/manuscript-source.zip")
    args = parser.parse_args()
    summary = {}
    if not args.skip_git:
        summary["git"] = validate_git()
        print("Git privacy checks passed", flush=True)
    summary["code"] = validate_code(args.code)
    print("Code archive and source hashes passed", flush=True)
    summary["data"] = validate_data(args.data)
    print("Data archive privacy and every manifest checksum passed", flush=True)
    if args.source.is_file():
        summary["manuscript_source"] = validate_source(args.source)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
