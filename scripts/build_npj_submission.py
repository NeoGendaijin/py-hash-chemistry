#!/usr/bin/env python3
"""Build clean/highlighted manuscripts, response documents and source archives."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "tex/Large-Hash-Chemistry"
OUT = ROOT / "dist/npj-submission"


def escape(text):
    substitutions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                     "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                     "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                     "<": r"\textless{}", ">": r"\textgreater{}"}
    return "".join(substitutions.get(c, c) for c in text)


def inline(text):
    parts = re.split(r"(`[^`]+`|https?://[^\s]+)", text)
    rendered = []
    for p in parts:
        if p.startswith("`") and p.endswith("`"):
            rendered.append(r"\nolinkurl{" + p[1:-1] + "}")
        elif p.startswith(("http://", "https://")):
            rendered.append(r"\url{" + p + "}")
        else:
            rendered.append(escape(p))
    return "".join(rendered)


def markdown_tex(source):
    lines = source.read_text().splitlines()
    output = [r"\documentclass[11pt,a4paper]{article}",
              r"\usepackage[margin=25mm]{geometry}",
              r"\usepackage[T1]{fontenc}", r"\usepackage{hyperref}",
              r"\hypersetup{hidelinks}", r"\setlength{\parindent}{0pt}",
              r"\setlength{\parskip}{6pt}", r"\emergencystretch=2em",
              r"\begin{document}"]
    paragraph = []

    def flush():
        if paragraph:
            output.append(inline(" ".join(paragraph)) + "\n")
            paragraph.clear()

    for line in lines:
        if not line.strip():
            flush()
        elif line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            command = {1: "section", 2: "subsection", 3: "subsubsection"}.get(level, "paragraph")
            output.append("\\" + command + "*{" + inline(line[level:].strip()) + "}")
        else:
            paragraph.append(line.strip())
    flush()
    output.append(r"\end{document}")
    return "\n".join(output) + "\n"


def sync_figures():
    pairs = [(ROOT / "results/npj_rev/schematics/outline-original.pdf",
              PAPER / "figures/family/rev1/outline-original.pdf"),
             (ROOT / "results/npj_rev/schematics/outline-original.svg",
              PAPER / "figures/family/rev1/outline-original.svg")]
    for name in ("P-D-final-num-runs.pdf", "P-D-final-size.pdf", "P-D-final-nctypes.pdf",
                 "fig5_heatmaps.pdf", "fig6_pairs.pdf"):
        pairs.append((ROOT / "results/npj_rev/ext1" / name, PAPER / "figures/ext1-dyadic/rev1" / name))
    for p in (ROOT / "results/npj_rev/ext1").glob("snap_*.png"):
        pairs.append((p, PAPER / "figures/ext1-dyadic/rev1" / p.name))
    for name in ("fig_schc_phenomenon.pdf", "fig_schc_mechanism.pdf"):
        pairs.append((ROOT / "results/npj_rev/ext2fig" / name, PAPER / "figures/ext2-scale/rev1" / name))
    for name in ("spatial_observables.pdf", "novelty_rate.pdf"):
        pairs.append((ROOT / "results/npj_rev/supplement" / name, PAPER / "figures/supplement" / name))
    pairs.append((ROOT / "results/npj_rev/ext2_tables/fig_horizon_40k.pdf",
                  PAPER / "figures/supplement/fig_horizon_40k.pdf"))
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def compile_tex(tectonic, source, destination, work):
    result = subprocess.run([tectonic, "-X", "compile", "--keep-logs", "--outdir", str(work),
                             str(source)], cwd=work, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    destination.with_suffix(".build.txt").write_text(result.stdout)
    if result.returncode:
        print(result.stdout[-12000:])
        result.check_returncode()
    log = work / (source.stem + ".log")
    if log.exists():
        text = log.read_text(errors="replace")
        if re.search(r"(?:undefined references|Citation .* undefined|Reference .* undefined)", text):
            raise RuntimeError(f"unresolved cross-reference in {log}")
        shutil.copyfile(log, destination.with_suffix(".log"))
    shutil.copyfile(work / (source.stem + ".pdf"), destination)
    print(f"Built {destination.name}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tectonic", default=shutil.which("tectonic"))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if not args.tectonic:
        raise RuntimeError("tectonic is required")
    args.out.mkdir(parents=True, exist_ok=True)
    sync_figures()
    manuscript = (PAPER / "sn-article-rev1.tex").read_text()
    if re.search(r"^[^%]*\\todo\{", manuscript, re.M):
        raise RuntimeError("unresolved manuscript TODO")
    with tempfile.TemporaryDirectory(prefix="npj-submission-") as tmp:
        work = Path(tmp)
        for pattern in ("*.cls", "*.sty", "*.bst", "*.bib"):
            for src in PAPER.glob(pattern):
                shutil.copyfile(src, work / src.name)
        for name in ("bst", "template"):
            src = PAPER / name
            if src.is_dir():
                shutil.copytree(src, work / name)
        (work / "figures").symlink_to(PAPER / "figures", target_is_directory=True)
        for label, highlight in (("highlighted", True), ("clean", False)):
            text = manuscript.replace("\n\\highlighttrue\n", "\n\\highlight" + ("true" if highlight else "false") + "\n")
            source = work / f"manuscript-{label}.tex"
            source.write_text(text)
            compile_tex(args.tectonic, source, args.out / f"manuscript-{label}.pdf", work)
        for name in ("response-to-reviewers", "cover-letter"):
            md = PAPER / f"review/{name}.md"
            source = work / f"{name}.tex"
            source.write_text(markdown_tex(md))
            compile_tex(args.tectonic, source, args.out / f"{name}.pdf", work)
            shutil.copyfile(md, args.out / md.name)
        source = work / "supplementary-information.tex"
        shutil.copyfile(PAPER / source.name, source)
        compile_tex(args.tectonic, source, args.out / "supplementary-information.pdf", work)
        source = work / "alife2026-erratum.tex"
        source.write_text(markdown_tex(ROOT / "docs/alife2026_erratum.md"))
        compile_tex(args.tectonic, source, args.out / "alife2026-erratum.pdf", work)

        used = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}",
                              manuscript + (PAPER / "supplementary-information.tex").read_text()))
        with zipfile.ZipFile(args.out / "manuscript-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(work / "manuscript-clean.tex", "manuscript.tex")
            archive.write(PAPER / "sn-article-rev1.tex", "manuscript-highlighted.tex")
            archive.write(PAPER / "supplementary-information.tex", "supplementary-information.tex")
            for p in sorted(work.iterdir()):
                if p.suffix in (".bib", ".cls", ".sty", ".bst"):
                    archive.write(p, p.name)
            for p in sorted(work.glob("bst/*.bst")):
                archive.write(p, p.relative_to(work).as_posix())
            for p in sorted(used):
                archive.write(PAPER / p, p)
    files = sorted(p for p in args.out.iterdir() if p.suffix in (".pdf", ".md", ".zip"))
    manifest = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files)
    (args.out / "SHA256SUMS").write_text(manifest)
    print(f"Submission artifacts: {args.out}")


if __name__ == "__main__":
    main()
