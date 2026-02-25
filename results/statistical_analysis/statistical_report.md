# Statistical Analysis Report

## Data Overview
- Sizes analyzed: [200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]
- Runs per size: L=200: 10, L=220: 10, L=240: 10, L=260: 10, L=280: 10, L=300: 10, L=320: 10, L=340: 10, L=360: 10, L=380: 10, L=400: 10

## Statistical Tests: L=300 vs L=320

### mean_size

**Final step values:**
- L=300: mean=2.6395, std=0.7690, median=2.3458
- L=320: mean=1855.0913, std=1949.5496, median=1366.5087
- Welch's t-test: t=-2.8506, p=1.91e-02
- Mann-Whitney U: U=18.0, p=1.73e-02
- Cohen's d: -1.2748

**Late-window values (last 25%):**
- L=300: mean=2.7322, std=0.8969
- L=320: mean=1741.7691, std=1789.9263
- Welch's t-test: t=-2.9147, p=1.72e-02
- Mann-Whitney U: U=25.0, p=6.40e-02
- Cohen's d: -1.3035

### max_size

**Final step values:**
- L=300: mean=13.9000, std=19.9722, median=5.5000
- L=320: mean=34110.9000, std=34102.6261, median=34005.5000
- Welch's t-test: t=-2.9995, p=1.50e-02
- Mann-Whitney U: U=19.0, p=2.06e-02
- Cohen's d: -1.3414

**Late-window values (last 25%):**
- L=300: mean=21.1570, std=32.4971
- L=320: mean=32024.9459, std=32565.4118
- Welch's t-test: t=-2.9483, p=1.63e-02
- Mann-Whitney U: U=20.0, p=2.57e-02
- Cohen's d: -1.3185

### mean_fitness

**Final step values:**
- L=300: mean=0.8720, std=0.1736, median=0.9634
- L=320: mean=0.6933, std=0.1908, median=0.6836
- Welch's t-test: t=2.0784, p=5.24e-02
- Mann-Whitney U: U=79.0, p=3.12e-02
- Cohen's d: 0.9295

**Late-window values (last 25%):**
- L=300: mean=0.8731, std=0.1681
- L=320: mean=0.6876, std=0.1906
- Welch's t-test: t=2.1896, p=4.22e-02
- Mann-Whitney U: U=84.0, p=1.13e-02
- Cohen's d: 0.9792

## Runaway Regime Analysis (mean_size > 100.0)

- L=200: 0/10 runs (0.0%)
- L=220: 0/10 runs (0.0%)
- L=240: 0/10 runs (0.0%)
- L=260: 0/10 runs (0.0%)
- L=280: 0/10 runs (0.0%)
- L=300: 0/10 runs (0.0%)
- L=320: 5/10 runs (50.0%)
- L=340: 7/10 runs (70.0%)
- L=360: 5/10 runs (50.0%)
- L=380: 5/10 runs (50.0%)
- L=400: 7/10 runs (70.0%)

## Threshold Crossing Times

### mean_size >= 10
- L=200: never crossed (0% of runs)
- L=220: mean=3182.0 +/- 0.0 steps (10% of runs)
- L=240: never crossed (0% of runs)
- L=260: mean=8768.7 +/- 5893.4 steps (30% of runs)
- L=280: mean=8436.8 +/- 7235.9 steps (40% of runs)
- L=300: mean=4748.0 +/- 1553.0 steps (20% of runs)
- L=320: mean=4261.6 +/- 2760.4 steps (50% of runs)
- L=340: mean=5027.3 +/- 4468.4 steps (70% of runs)
- L=360: mean=4713.0 +/- 2723.4 steps (50% of runs)
- L=380: mean=3608.2 +/- 3031.9 steps (50% of runs)
- L=400: mean=8510.7 +/- 3922.5 steps (70% of runs)

### mean_size >= 50
- L=200: never crossed (0% of runs)
- L=220: never crossed (0% of runs)
- L=240: never crossed (0% of runs)
- L=260: never crossed (0% of runs)
- L=280: never crossed (0% of runs)
- L=300: never crossed (0% of runs)
- L=320: mean=9464.8 +/- 4079.9 steps (50% of runs)
- L=340: mean=7246.7 +/- 4772.0 steps (70% of runs)
- L=360: mean=5918.2 +/- 3409.7 steps (50% of runs)
- L=380: mean=4555.2 +/- 2725.1 steps (50% of runs)
- L=400: mean=9337.3 +/- 3807.6 steps (70% of runs)

### mean_size >= 100
- L=200: never crossed (0% of runs)
- L=220: never crossed (0% of runs)
- L=240: never crossed (0% of runs)
- L=260: never crossed (0% of runs)
- L=280: never crossed (0% of runs)
- L=300: never crossed (0% of runs)
- L=320: mean=9497.0 +/- 4079.6 steps (50% of runs)
- L=340: mean=7278.7 +/- 4774.0 steps (70% of runs)
- L=360: mean=5951.2 +/- 3410.8 steps (50% of runs)
- L=380: mean=4589.2 +/- 2726.8 steps (50% of runs)
- L=400: mean=9369.6 +/- 3807.6 steps (70% of runs)

### mean_size >= 500
- L=200: never crossed (0% of runs)
- L=220: never crossed (0% of runs)
- L=240: never crossed (0% of runs)
- L=260: never crossed (0% of runs)
- L=280: never crossed (0% of runs)
- L=300: never crossed (0% of runs)
- L=320: mean=9607.2 +/- 4087.4 steps (50% of runs)
- L=340: mean=7383.1 +/- 4771.1 steps (70% of runs)
- L=360: mean=6051.4 +/- 3413.4 steps (50% of runs)
- L=380: mean=4689.4 +/- 2722.9 steps (50% of runs)
- L=400: mean=9474.1 +/- 3804.6 steps (70% of runs)

### mean_size >= 1000
- L=200: never crossed (0% of runs)
- L=220: never crossed (0% of runs)
- L=240: never crossed (0% of runs)
- L=260: never crossed (0% of runs)
- L=280: never crossed (0% of runs)
- L=300: never crossed (0% of runs)
- L=320: mean=9676.8 +/- 4086.2 steps (50% of runs)
- L=340: mean=7451.0 +/- 4771.6 steps (70% of runs)
- L=360: mean=6127.4 +/- 3417.2 steps (50% of runs)
- L=380: mean=4764.8 +/- 2737.1 steps (50% of runs)
- L=400: mean=9546.3 +/- 3797.5 steps (70% of runs)
