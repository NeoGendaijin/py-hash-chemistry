# npj Complexity revision 1

Code and processed data for I. Horiguchi and H. Sayama,
*Hash Chemistry models for evolutionary growth of complexity*.

Both extensions have been recomputed after correcting winner selection, replacing
the degenerate XOR surrogate and restoring the intended dyadic pair rule.
The size scan has 100 seeds at each of 19 sizes; its stochastic onset rises most
steeply near L=220-240 at 20,000 steps. Earlier periodic controls are withdrawn.
The ALIFE simulator is preserved at tag alife2026-camera-ready.

The manuscript explicitly distinguishes the archived JAX corner exception from
the corrected normal JAX write path. Paper reproduction drivers select
legacy_corner=True. The C size scan is unaffected.

## Assets

- npj-complexity-data.tar.gz: corrected per-run trajectories, summaries,
  validation outputs, derived statistics and grid snapshots.
- npj-complexity-data.tar.gz.sha256: archive checksum.
- PROVENANCE_npj.json: code commit, source hashes, package versions, seeds
  and inventory, also embedded in the archive.

Private Mathematica attachments and superseded periodic data are excluded.
See docs/npj_data_README.md for the complete inventory and definitions.

## Reproduce

```bash
python -m pip install -e ".[paper]" -c docs/npj-analysis-constraints.txt
tar -xzf npj-complexity-data.tar.gz
sha256sum -c results/MANIFEST_npj.sha256
python scripts/npj_rev_ext1.py
python scripts/npj_rev_ext2_tables.py
python scripts/npj_rev_ext2_figs.py --no-provisional
python scripts/npj_revision_schematics.py
python scripts/npj_revision_supplement.py
```

This text is prepared for release publication. It does not assert that an ALIFE
correction notice has already been accepted or published.
