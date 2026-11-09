# Structural Cellular Hash Chemistry

This branch contains the revision of I. Horiguchi and H. Sayama,
*Hash Chemistry models for evolutionary growth of complexity* (npj Complexity).
It includes both the spatial-dyadic extension and the size scan. The planned
release tag is `npj-complexity-rev1`; publication of that release is a separate step.
The ALIFE 2026 implementation is preserved at tag `alife2026-camera-ready`.

## Corrections and model versions

During revision we found and disclosed three implementation problems.

- The earlier JAX winner rule let the higher score win only three quarters of
  non-tied contests. The corrected rule always selects the higher score and
  breaks exact ties uniformly.
- The earlier XOR score ignores permutations of cell types within a fixed
  shape. The revised score mixes each cell's type and relative coordinates
  before a wrapping sum.
- A typographical error in the earlier Mathematica dyadic rule prevented the
  canonical ordering of contestants from being restored. The intended rule
  fixes the winner of each pair independently of sampling order.

Both extensions were recomputed. The size scan now uses 100 seeds per size;
its onset is stochastic and the probability rises most steeply near L=220-240
within 20,000 steps. The former L=300-320 threshold is not retained. Periodic
controls are withdrawn because periodic scoring and copying were not valid.

The primary JAX sweep, comparisons and snapshots now use the corrected corner
write and are stored under `results/corner_fixed/`. Historical runs retain the
scatter exception at (0,0) and remain separate; `legacy_corner=True` selects
that behavior only for historical reproduction. The C engine never had this
exception. Adaptive batches retain each run's state and PRNG and split out
large runs for efficiency; GPU checks establish exact agreement in tested cases.

An ALIFE correction notice has been prepared; it has not yet been submitted or
published. Correspondence and submission drafts are not public benchmark assets.

## Install

The recorded experiments used Python 3.10; use that version for reproduction. The CPU engine also needs GCC (or a compatible
`gcc` executable). Figure generation runs on the CPU.

```bash
uv venv --python 3.10
uv pip install -e ".[paper]"
```

For an analysis environment matching the revision, use
[the exact analysis versions](docs/npj-analysis-constraints.txt):

```bash
uv pip install -e ".[paper]" -c docs/npj-analysis-constraints.txt
```

## Reproduce the revised figures and tables

Extract the prepared/released archive at the repository root, then run:

```bash
tar -xzf npj-complexity-data.tar.gz
sha256sum -c results/MANIFEST_npj.sha256
uv run --no-sync python scripts/npj_rev_ext1.py
uv run --no-sync python scripts/npj_rev_ext2_tables.py
uv run --no-sync python scripts/npj_rev_ext2_figs.py --no-provisional
uv run --no-sync python scripts/npj_revision_schematics.py
uv run --no-sync python scripts/npj_revision_supplement.py
```

Outputs are in `results/npj_rev/`. These commands reproduce Figures 1 and
5-9 and Table 2 without access to the manuscript checkout or private email
attachments. The original reference figures remain in the manuscript repository.
The archive inventory, seeds, sampling conventions, validation inputs and
SHA-256 source provenance are described in [the data guide](docs/npj_data_README.md).

## Simulation and validation

```python
from hash_chemistry.simulation import SCHCParams
from hash_chemistry.fast import run

params = SCHCParams()  # original winner rule, mix score, open boundaries
rows, final_grid = run(L=100, seed=0, steps=100)
```

Two active cells are sampled with replacement; each selects its 8-connected
component. The winner's bounding box plus a one-cell margin is centered on the
loser and copied from the pre-contest grid. Death and mutation apply to every
copied cell, including empty cells. Open boundaries clip the copy. Sampling the
same component twice produces a self-copy with mutation. A generation accumulates
winner-size / active-cell-count until it reaches one.

The corrected simulation drivers and parameters are:

- Extension I: `scripts/run_spatial_dyadic_sweep.py` (JAX/GPU;
  corrected corner writes; historical behavior requires an explicit flag).
- Extension II: `scripts/run_scan_cpu.py` and
  `scripts/run_npj_recompute.sh` (C/CPU; full reruns are optional).
- Rule checks: `JAX_PLATFORMS=cpu uv run --no-sync python scripts/validate_model_rules.py`.
- Extension I reporting checks: `uv run --no-sync python scripts/validate_ext1_observables.py`.
- Distribution checks: `uv run --no-sync python scripts/validate_cpu_engine.py`.

JAX and C use different random generators. Equal seed numbers do not identify
the same run across engines. Archived ALIFE behavior requires
`SCHCParams(winner_rule="legacy", score="xor", legacy_corner=True)`.
Older exploratory runners are not the revision reproduction entry points.
`SCHCParams(periodic=True)` raises an error until periodic rules are validated.

## Preparing release assets

```bash
uv run --no-sync python scripts/package_npj_data.py
```

This builds `dist/npj-complexity-data.tar.gz`, its checksum and provenance.
It excludes obsolete periodic data and privately supplied Mathematica attachments.
The public code archive also excludes gitignored material, and the data archive
omits working Markdown reports. The public data guide is included explicitly.
Code/data publication and manuscript submission must use the same frozen version.
