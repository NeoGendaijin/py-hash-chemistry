## npj Complexity, revision 1

Code and data for the SCHC finite-size transition study ("second extension") in
I. Horiguchi and H. Sayama, *Hash Chemistry: Minimal Models for Evolutionary
Growth of Complexity*, npj Complexity.

The code for the ALIFE 2026 paper is on branch `main` (tag `alife2026-camera-ready`).

### Assets
- `npj-complexity-data.tar.gz`: per-run trajectories, per-size summaries,
  derived statistics and grid snapshots behind every SCHC figure and table
  of the paper. The contents and file formats are described in
  `docs/npj_data_README.md` (also included inside the archive).
- `npj-complexity-data.tar.gz.sha256`: checksum of the archive.

### Reproduce
```bash
git clone --branch npj-complexity-rev1 https://github.com/NeoGendaijin/py-hash-chemistry
cd py-hash-chemistry && pip install -e ".[paper]"
tar -xzf /path/to/npj-complexity-data.tar.gz
python scripts/transition_uncertainty.py     # Table 2 with 95% intervals
python scripts/generate_fig_schc_figs.py     # figures -> results/figures_npj/
```
