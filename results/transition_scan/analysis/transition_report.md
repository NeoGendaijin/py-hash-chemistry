# Transition Scan Report (L=200..400)

## Context
This report analyzes the mid-range spatial sizes to locate the qualitative transition noted between L=200 and L=400, using 10 runs per size and 20,000 steps.

## Windows
- early: 2000..5000 steps
- mid: 5000..10000 steps
- late: 15000..20000 steps
- corr: 2000..10000 steps

## Key Findings
- Mean component size stays ~2-3 for L<=300, then jumps to ~1,800+ at L=320.
- The first crossing of mean_size >= 10/100/1000 appears only for L>=320.
- Mean size vs fitness shows strong negative correlation for L>=320.
- Mean fitness tends to decrease from early to late windows for L>=320.

## Per-Size Summary (final step and transition markers)

| L | mean_size_end | max_size_end | late/early (mean_size) | t(mean_size>=10) | t(mean_size>=100) | t(mean_size>=1000) | corr(size,fitness) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | 2.936 | 6.8 | 0.99 | nan | nan | nan | 0.20 |
| 220 | 2.868 | 9.6 | 0.96 | nan | nan | nan | 0.07 |
| 240 | 2.597 | 6.6 | 0.99 | nan | nan | nan | 0.27 |
| 260 | 3.224 | 15.6 | 1.01 | nan | nan | nan | -0.21 |
| 280 | 2.768 | 12.4 | 0.92 | nan | nan | nan | -0.43 |
| 300 | 2.639 | 13.9 | 1.06 | nan | nan | nan | -0.39 |
| 320 | 1855.091 | 34110.9 | 12.13 | 3735 | 3911 | 9400 | -0.76 |
| 340 | 3189.792 | 53997.2 | 6.19 | 1925 | 2122 | 4752 | -0.79 |
| 360 | 1677.786 | 43175.7 | 3.63 | 995 | 1182 | 4333 | -0.83 |
| 380 | 1941.392 | 48053.4 | 2.80 | 1350 | 1530 | 3809 | -0.56 |
| 400 | 2656.930 | 74601.0 | 35.81 | 4100 | 4295 | 6429 | -0.88 |

## Figures
- Mean size (final) vs L: `results/transition_scan/analysis/mean_size_end_vs_L.png`
- Max size (final) vs L: `results/transition_scan/analysis/max_size_end_vs_L.png`
- Late/Early mean size ratio vs L: `results/transition_scan/analysis/mean_size_late_over_early_vs_L.png`
- Size/Fitness correlation vs L: `results/transition_scan/analysis/corr_size_fitness_vs_L.png`
- Late/Early mean fitness ratio vs L: `results/transition_scan/analysis/mean_fitness_late_over_early_vs_L.png`
- Time to mean size thresholds: `results/transition_scan/analysis/time_to_mean_size_thresholds.png`

## Notes
The sharp jump between L=300 and L=320 is consistent with a transition from local fitness optimization to size-dominant competition, as suggested in the earlier discussion with Prof. Sayama.
