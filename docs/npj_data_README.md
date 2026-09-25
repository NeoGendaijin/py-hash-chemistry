# Data for the SCHC finite-size transition study (npj Complexity)

This archive holds the processed simulation outputs behind every figure and
table of the second extension ("Extension II": the scale-controlled transition
in Structural Cellular Hash Chemistry) of

> I. Horiguchi and H. Sayama, *Hash Chemistry: Minimal Models for Evolutionary
> Growth of Complexity*, npj Complexity.

It belongs to the code release tagged `npj-complexity-rev1` of
<https://github.com/NeoGendaijin/py-hash-chemistry> (branch `npj-complexity`).
Extract it at the root of that checkout; everything lands under `results/`:

```bash
tar -xzf npj-complexity-data.tar.gz
sha256sum -c results/MANIFEST_npj.sha256     # optional integrity check
```

## Contents

| Directory | Experiment (paper Methods) | Runs | Steps | Sampling | Used in |
|---|---|---|---|---|---|
| `transition_scan/L{200..400}/` | coarse size scan, open boundaries, L = 200, 220, ..., 400 | 10 per L | 20,000 | every step | Table 2, Fig. 7c–d, Fig. 8b, 8e |
| `fine_transition_scan/L{300..320}/` | fine size scan, open boundaries, L = 300, 302, ..., 320 | 10 per L | 20,000 | every 10 steps | Fig. 7c–d, text |
| `boundary_control/periodic_v2/L{200..360}/` | periodic (toroidal) boundaries, L = 200, 240, 280, 300, 320, 360 | 5 per L | up to 5,000 | every 10 steps | Fig. 8a |
| `large_space/L{100,200,400}/` | long-run comparison | 100 per L | 20,000 | every step | Fig. 7e, text |
| `param_sensitivity_mu/mu_{value}/L{L}/` | mutation-rate sweep, L = 200, 300, 320, 400 | 5 per cell | 10,000 | every step | Fig. 8c |
| `param_sensitivity_death/death_prob_{value}/L{L}/` | death-probability sweep, L = 200, 300, 320, 400 | 5 per cell | 10,000 | every step | Fig. 8d |
| `statistical_analysis/` | derived statistics (tests, 95% intervals) | | | | Table 2, text |
| `gifs/configs_L200_seed0/`, `gifs/configs_L400_seed8/` | grid snapshots (`.npy`) | | | | Fig. 7a–b |

The open-boundary scans use the default parameters (k = 1000, n = 10,
mu = 0.002/0.999, p_death = 0.001). Periodic runs stop at the first sample at
which the mean component size reaches 2,000 cells, where runaway is
unambiguous; runs that never reach it continue to step 5,000.

## File formats

**Per-run trajectories** `.../runs/seed_{s}.csv` (seed = run index), one row per
sampled step:

| Column | Meaning |
|---|---|
| `step` | simulation step (one step = contests until the elapsed time reaches 1) |
| `max_fitness`, `mean_fitness` | maximum / mean **hash score** over all connected components (the paper's hash score; the column name is historical) |
| `max_size`, `mean_size` | size in cells of the largest component / mean component size (8-connectivity) |
| `cum_cell_types` | cumulative number of distinct cell types observed so far |
| `cum_pattern_types` | cumulative number of distinct replicated patterns (translation-normalized shape plus cell types) |

**Per-size summaries** `summary.csv`: for every step, the mean and standard
deviation over runs of each column above (`<column>_mean`, `<column>_std`).

**Analyses**: `transition_scan/analysis/transition_analysis.csv` and
`fine_transition_scan/analysis/fine_transition_analysis.csv` hold the
per-size quantities of Table 2; `param_sensitivity_*/sensitivity_summary_*.csv`
give the final mean size shown in the phase diagrams; `statistical_analysis/`
contains the 95% intervals and tests reported in the text
(`transition_uncertainty.csv`, `fine_transition_uncertainty.csv`,
`per_size_summary.csv`).

**Snapshots** `gifs/configs_L{L}_seed{s}/config_step{t}.npy`: integer arrays of
shape L x L; 0 = empty cell, 1..k = cell type.

## Regenerating the figures and tables

```bash
python scripts/transition_uncertainty.py    # Table 2 with 95% intervals
python scripts/generate_fig_schc_figs.py    # Figs. 7 and 8
```

Figures are written to `results/figures_npj/` (or to `$SCHC_FIG_DIR` if set).
The raw simulations can be rerun with `scripts/run_transition_scan.sh`,
`scripts/run_fine_transition_scan.sh`, `scripts/run_large_space.sh`,
`scripts/param_sensitivity.py`, and, for the periodic control,

```bash
python scripts/run_scan_pool.py --periodic --sizes 200 240 280 300 320 360 \
    --seeds 0-4 --steps 5000 --stop-mean-size 2000 \
    --output-root results/boundary_control/periodic_v2
```

(GPU recommended; see each script's `--help`).
