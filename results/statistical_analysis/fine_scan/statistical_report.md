# Statistical Analysis Report

## Data Overview
- Sizes analyzed: [300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320]
- Runs per size: L=300: 10, L=302: 10, L=304: 10, L=306: 10, L=308: 10, L=310: 10, L=312: 10, L=314: 10, L=316: 10, L=318: 10, L=320: 10

## Statistical Tests: L=308 vs L=312

### mean_size

**Final step values:**
- L=308: mean=1020.2718, std=1556.7559, median=2.9947
- L=312: mean=1134.1660, std=1762.0332, median=3.3085
- Welch's t-test: t=-0.1453, p=8.86e-01
- Mann-Whitney U: U=56.0, p=6.78e-01
- Cohen's d: -0.0650

**Late-window values (last 25%):**
- L=308: mean=1062.0276, std=1618.6439
- L=312: mean=1105.0373, std=1686.6083
- Welch's t-test: t=-0.0552, p=9.57e-01
- Mann-Whitney U: U=54.0, p=7.91e-01
- Cohen's d: -0.0247

### max_size

**Final step values:**
- L=308: mean=18959.3000, std=28948.6062, median=8.5000
- L=312: mean=19516.3000, std=29791.6003, median=11.0000
- Welch's t-test: t=-0.0402, p=9.68e-01
- Mann-Whitney U: U=50.5, p=1.00e+00
- Cohen's d: -0.0180

**Late-window values (last 25%):**
- L=308: mean=18856.1431, std=28791.4570
- L=312: mean=19469.4639, std=29715.5633
- Welch's t-test: t=-0.0445, p=9.65e-01
- Mann-Whitney U: U=53.0, p=8.50e-01
- Cohen's d: -0.0199

### mean_fitness

**Final step values:**
- L=308: mean=0.7459, std=0.1680, median=0.8331
- L=312: mean=0.7799, std=0.2117, median=0.8579
- Welch's t-test: t=-0.3775, p=7.10e-01
- Mann-Whitney U: U=41.0, p=5.21e-01
- Cohen's d: -0.1688

**Late-window values (last 25%):**
- L=308: mean=0.7420, std=0.1671
- L=312: mean=0.7721, std=0.2204
- Welch's t-test: t=-0.3260, p=7.48e-01
- Mann-Whitney U: U=39.0, p=4.27e-01
- Cohen's d: -0.1458

## Runaway Regime Analysis (mean_size > 100.0)

- L=300: 0/10 runs (0.0%)
- L=302: 3/10 runs (30.0%)
- L=304: 5/10 runs (50.0%)
- L=306: 2/10 runs (20.0%)
- L=308: 3/10 runs (30.0%)
- L=310: 2/10 runs (20.0%)
- L=312: 3/10 runs (30.0%)
- L=314: 1/10 runs (10.0%)
- L=316: 0/10 runs (0.0%)
- L=318: 3/10 runs (30.0%)
- L=320: 5/10 runs (50.0%)

## Threshold Crossing Times

### mean_size >= 10
- L=300: mean=8440.0 +/- 5240.0 steps (20% of runs)
- L=302: mean=7948.6 +/- 5260.1 steps (70% of runs)
- L=304: mean=6360.0 +/- 5371.9 steps (70% of runs)
- L=306: mean=9396.7 +/- 3839.6 steps (30% of runs)
- L=308: mean=5325.0 +/- 2993.4 steps (40% of runs)
- L=310: mean=7641.7 +/- 5976.7 steps (60% of runs)
- L=312: mean=4765.0 +/- 1785.9 steps (40% of runs)
- L=314: mean=3440.0 +/- 650.0 steps (20% of runs)
- L=316: mean=10260.0 +/- 9070.0 steps (20% of runs)
- L=318: mean=7583.3 +/- 5480.9 steps (60% of runs)
- L=320: mean=4264.0 +/- 2759.3 steps (50% of runs)

### mean_size >= 50
- L=300: never crossed (0% of runs)
- L=302: mean=8570.0 +/- 4338.1 steps (30% of runs)
- L=304: mean=12194.0 +/- 5016.1 steps (50% of runs)
- L=306: mean=10365.0 +/- 5275.0 steps (20% of runs)
- L=308: mean=11646.7 +/- 2424.9 steps (30% of runs)
- L=310: mean=7725.0 +/- 2595.0 steps (20% of runs)
- L=312: mean=8593.3 +/- 3070.6 steps (30% of runs)
- L=314: mean=5320.0 +/- 0.0 steps (10% of runs)
- L=316: never crossed (0% of runs)
- L=318: mean=8243.3 +/- 2499.1 steps (30% of runs)
- L=320: mean=9468.0 +/- 4079.6 steps (50% of runs)

### mean_size >= 100
- L=300: never crossed (0% of runs)
- L=302: mean=8603.3 +/- 4333.4 steps (30% of runs)
- L=304: mean=12228.0 +/- 5016.2 steps (50% of runs)
- L=306: mean=10395.0 +/- 5275.0 steps (20% of runs)
- L=308: mean=11676.7 +/- 2424.9 steps (30% of runs)
- L=310: mean=7755.0 +/- 2595.0 steps (20% of runs)
- L=312: mean=8623.3 +/- 3070.6 steps (30% of runs)
- L=314: mean=5360.0 +/- 0.0 steps (10% of runs)
- L=316: never crossed (0% of runs)
- L=318: mean=8276.7 +/- 2499.4 steps (30% of runs)
- L=320: mean=9500.0 +/- 4079.2 steps (50% of runs)

### mean_size >= 500
- L=300: never crossed (0% of runs)
- L=302: mean=8716.7 +/- 4336.3 steps (30% of runs)
- L=304: mean=12336.0 +/- 5010.1 steps (50% of runs)
- L=306: mean=10510.0 +/- 5280.0 steps (20% of runs)
- L=308: mean=11780.0 +/- 2422.1 steps (30% of runs)
- L=310: mean=7860.0 +/- 2600.0 steps (20% of runs)
- L=312: mean=8730.0 +/- 3064.2 steps (30% of runs)
- L=314: mean=5450.0 +/- 0.0 steps (10% of runs)
- L=316: never crossed (0% of runs)
- L=318: mean=8383.3 +/- 2499.1 steps (30% of runs)
- L=320: mean=9612.0 +/- 4082.7 steps (50% of runs)

### mean_size >= 1000
- L=300: never crossed (0% of runs)
- L=302: mean=8793.3 +/- 4326.5 steps (30% of runs)
- L=304: mean=12410.0 +/- 5008.1 steps (50% of runs)
- L=306: mean=10585.0 +/- 5295.0 steps (20% of runs)
- L=308: mean=11850.0 +/- 2423.1 steps (30% of runs)
- L=310: mean=7935.0 +/- 2585.0 steps (20% of runs)
- L=312: mean=8813.3 +/- 3067.0 steps (30% of runs)
- L=314: mean=5530.0 +/- 0.0 steps (10% of runs)
- L=316: never crossed (0% of runs)
- L=318: mean=8450.0 +/- 2507.8 steps (30% of runs)
- L=320: mean=9680.0 +/- 4085.9 steps (50% of runs)
