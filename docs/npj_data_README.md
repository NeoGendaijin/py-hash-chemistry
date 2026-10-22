# Data for the npj Complexity revision

This archive accompanies I. Horiguchi and H. Sayama, *Hash Chemistry models for
evolutionary growth of complexity*. It contains the corrected simulations for
both extensions, validation outputs and derived statistics. It replaces the
September archive, which contained the earlier simulator and withdrawn periodic
controls. The target code release is `npj-complexity-rev1`.

Extract at the root of the accompanying code checkout:

```bash
tar -xzf npj-complexity-data.tar.gz
sha256sum -c results/MANIFEST_npj.sha256
```

The manifest covers every data file and the embedded README and provenance.
`results/PROVENANCE_npj.json` records code commit, working-tree status,
source SHA-256 hashes, analysis versions, seeds and an inventory. Source hashes
identify the reproduction implementation at packaging; they do not invent an
experiment-time commit for formerly uncommitted work.

## Main datasets

| Directory below results/ | Content | Seeds / horizon | Manuscript use |
|---|---|---|---|
| spatial_dyadic/intended | JAX; 60 (D,P) conditions, L=100; intended dyadic rule | 0-29 / 2,000 steps | Extension I, Figs. 5-7 |
| spatial_dyadic/as_coded | JAX; 50 P>0 conditions; earlier Mathematica dyadic behavior | 0-29 / 2,000 | comparison and validation |
| spatial_dyadic_snapshots/intended | four conditions, configurations at 1,500, 1,750 and 2,000 steps | 0-7 | Fig. 7 and selection audit |
| cpu_scan_full/original_mix | C; 19 L values, no growth-based early stopping | 0-99 / 20,000 | Extension II, Table 2, Figs. 8-9 |
| cpu_novelty/original_mix/L200 | C; every-step metrics | 0-39 / 20,000 | Fig. 8e, novelty-rate check |
| cpu_sensitivity | C; mutation/death sweeps at six L values | 0-29 / 10,000 | Fig. 9c,d |
| cpu_scan_40k/original_mix | C; L=240,320,400; early stop at mean size 2,000 | 0-99 / up to 40,000 | longer-horizon first passages |
| cpu_snapshots | C; L=200 seed 0 and L=400 seed 5 at figure times | stated seeds | Fig. 8a,b |
| cpu_scan/legacy_xor | C; earlier winner rule and XOR score | 0-99 / up to 20,000 | Fig. 8d grey comparison |
| cpu_scan/original_mix | earlier stopped version of the corrected scan | 0-99 / up to 20,000 | prefix/determinism validation only |
| npj_rev | per-run summaries, confidence intervals, tests and tables | derived | all revised results |

The 19 size-scan values are 100,120,140,160,180,200,210,220,230,240,250,
260,280,300,320,340,360,380,400.
Corrected runs use winner_rule=original, score=mix, k=1000, n=10,
mu=0.002/0.999 and death_prob=0.001 unless a sensitivity condition states otherwise.
The default sensitivity rows use the conditional mu=0.002/0.999, not exactly
0.002. The extra exact-mu=0.002 runs are included for provenance but do not
replace the default row.

## Sampling and observables

C trajectory CSVs have columns step, max_fitness, mean_fitness, max_size,
mean_size, cum_cell_types, cum_pattern_types, n_active and n_components.

- Historical fitness column names denote hash scores, not realized fitness.
- Sizes and scores are measured across current 8-connected components.
- C cumulative pattern counts include components present at sampled times,
  whether or not those components replicated. Counts use a 64-bit pattern
  identifier (two mixed 32-bit hashes), up to translation.
- In the legacy/xor C comparison the score columns are alternative mix-score
  probes; competition still uses XOR. Only its size/runaway results are used.
- C trajectories are sampled every ten steps, except the every-step novelty
  dataset. Extinct runs end at extinction; analysis pads their sizes with zero.
- The 40,000-step and legacy scans stop after mean size reaches 2,000.
  They support first-passage statistics, not subsequent persistence claims.
- Snapshots are integer arrays; zero is empty and positive entries are cell types.

Extension I CSVs record winner-event statistics, including mean_rep_size,
max_rep_size, cum_pattern_types, n_events, n_active and log_bytes_cum.
Row t records events of step t+1; n_active is the population before those events.
Thus the last row numbered 1999 contains the cumulative event novelty after
2,000 steps. Its pattern count concerns replicated winners and differs from the
population-snapshot count of Extension II. Paper scripts reproduce the
specified success filter and also report sensitivity to exclusions.

All archived JAX runs retain the historical corner scatter through
legacy_corner=True. Invalid padded writes could suppress a valid update at
(0,0). The release's normal JAX path fixes that write; changing the full
spatial-dyadic sweep to the new path has not been tested. The C scan has no
corner exception. Neither dataset contains a valid periodic-boundary control.

## Validation data

The original_comparison directory holds the legacy, winner-only-corrected and
winner-plus-mix JAX arms (100 seeds each) and the 35-comparison summary.
The original Mathematica *Series.dat inputs are publicly available at
https://github.com/hsayama/Structural-Cellular-Hash-Chemistry.
Use scripts/analyze_original_comparison.py --original-dir /path/to/that/checkout
to repeat those reference comparisons.

transition_fixed, transition_fixed_check, cpu_scan_check, cpu_scan_L100 and
large_space/L200,L400 contain the JAX/C distribution-comparison trajectories.
JAX and C seed numbers are not aligned across engines.

The privately supplied per-run spatial-dyadic Mathematica CSV is not included.
It is available from H. Sayama on reasonable request. Its derived comparison
statistics are in spatial_dyadic/stats. If that private CSV is absent,
npj_rev_ext1.py still regenerates every main Extension I figure and all JAX
statistics, and skips the optional Mathematica regression.
No correspondence or private attachment is included.

## Reproduction

```bash
python scripts/npj_rev_ext1.py
python scripts/npj_rev_ext2_tables.py
python scripts/npj_rev_ext2_figs.py --no-provisional
python scripts/npj_revision_schematics.py
python scripts/npj_revision_supplement.py
JAX_PLATFORMS=cpu python scripts/validate_model_rules.py
python scripts/validate_cpu_engine.py
```

Output: results/npj_rev/ext1, ext2_tables, ext2fig, schematics and supplement.
The source scripts never overwrite manuscript TeX or original figure assets.
