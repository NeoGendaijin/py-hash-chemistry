#!/usr/bin/env bash
set -euo pipefail

# V4: One process per GPU, sequential. Optimal for GPU-bound JAX workload.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

FINE_OUT="$ROOT/results/fine_transition_scan"
SNAP_OUT="$ROOT/results/snapshots"
MU_OUT="$ROOT/results/param_sensitivity_mu"
DEATH_OUT="$ROOT/results/param_sensitivity_death"
ALT_OUT="$ROOT/results/alt_hash"
STATS_OUT="$ROOT/results/statistical_analysis"

mkdir -p "$FINE_OUT" "$SNAP_OUT" "$MU_OUT" "$DEATH_OUT" "$ALT_OUT" "$STATS_OUT"

# One process per GPU, running its assigned tasks sequentially
# Each fine scan seed: ~2 min. Each param sense seed: ~1 min.

# GPU 0: Fine scan L=300(10),302(10),304(10) = 30 seeds (~60 min)
#         Then param mu all values for L=200 (4 vals × 5 seeds = 20 runs, ~20 min)
#         Then param mu all values for L=300 (4 vals × 5 seeds = 20 runs, ~20 min)
# Total: ~100 min
gpu0_work() {
  export CUDA_VISIBLE_DEVICES=0
  echo "[GPU0] Fine scan L=300,302,304"
  for size in 300 302 304; do
    for seed in $(seq 0 9); do
      "$PY" scripts/fast_scan.py --size $size --seed $seed --steps 20000 \
        --sample-interval 10 --output-root "$FINE_OUT" --progress-interval 5000
    done
  done
  echo "[GPU0] Param mu L=200,300"
  "$PY" scripts/param_sensitivity.py --param mu --values 0.001 0.002 0.005 0.01 \
    --sizes 200 300 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$MU_OUT" --progress-interval 2000
  echo "[GPU0] DONE"
}

# GPU 1: Fine scan L=306(10),308(10),310(10) = 30 seeds (~60 min)
#         Then param mu for L=320,400 (~40 min)
gpu1_work() {
  export CUDA_VISIBLE_DEVICES=1
  echo "[GPU1] Fine scan L=306,308,310"
  for size in 306 308 310; do
    for seed in $(seq 0 9); do
      "$PY" scripts/fast_scan.py --size $size --seed $seed --steps 20000 \
        --sample-interval 10 --output-root "$FINE_OUT" --progress-interval 5000
    done
  done
  echo "[GPU1] Param mu L=320,400"
  "$PY" scripts/param_sensitivity.py --param mu --values 0.001 0.002 0.005 0.01 \
    --sizes 320 400 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$MU_OUT" --progress-interval 2000
  echo "[GPU1] DONE"
}

# GPU 2: Fine scan L=312(10),316(10),320(10) = 30 seeds (~60 min)
#         Then snapshots L=200 (~12 min)
#         Then param death L=200,300 (~40 min)
gpu2_work() {
  export CUDA_VISIBLE_DEVICES=2
  echo "[GPU2] Fine scan L=312,316,320"
  for size in 312 316 320; do
    for seed in $(seq 0 9); do
      "$PY" scripts/fast_scan.py --size $size --seed $seed --steps 20000 \
        --sample-interval 10 --output-root "$FINE_OUT" --progress-interval 5000
    done
  done
  echo "[GPU2] Snapshots L=200"
  "$PY" scripts/generate_snapshots.py --sizes 200 --seed 0 --steps 20000 \
    --output-dir "$SNAP_OUT" --progress-interval 2000
  echo "[GPU2] Param death L=200,300"
  "$PY" scripts/param_sensitivity.py --param death_prob --values 0.0005 0.001 0.005 0.01 \
    --sizes 200 300 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$DEATH_OUT" --progress-interval 2000
  echo "[GPU2] DONE"
}

# GPU 3: Fine scan L=314(10),318(10) = 20 seeds (~40 min)
#         Then snapshots L=400 (~12 min)
#         Then param death L=320,400 (~40 min)
#         Then alt hash 4 sizes × 5 reps (~20 min)
gpu3_work() {
  export CUDA_VISIBLE_DEVICES=3
  echo "[GPU3] Fine scan L=314,318"
  for size in 314 318; do
    for seed in $(seq 0 9); do
      "$PY" scripts/fast_scan.py --size $size --seed $seed --steps 20000 \
        --sample-interval 10 --output-root "$FINE_OUT" --progress-interval 5000
    done
  done
  echo "[GPU3] Snapshots L=400"
  "$PY" scripts/generate_snapshots.py --sizes 400 --seed 0 --steps 20000 \
    --output-dir "$SNAP_OUT" --progress-interval 2000
  echo "[GPU3] Param death L=320,400"
  "$PY" scripts/param_sensitivity.py --param death_prob --values 0.0005 0.001 0.005 0.01 \
    --sizes 320 400 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$DEATH_OUT" --progress-interval 2000
  echo "[GPU3] Alt hash"
  "$PY" scripts/alt_hash_test.py --sizes 200 300 320 400 --runs 5 --steps 10000 \
    --seed0 0 --output-root "$ALT_OUT" --progress-interval 2000
  echo "[GPU3] DONE"
}

echo "Launching 4 GPU workers (1 per GPU)..."
gpu0_work > "$ROOT/results/gpu0_v4.log" 2>&1 &
gpu1_work > "$ROOT/results/gpu1_v4.log" 2>&1 &
gpu2_work > "$ROOT/results/gpu2_v4.log" 2>&1 &
gpu3_work > "$ROOT/results/gpu3_v4.log" 2>&1 &

echo "Waiting for all GPUs..."
wait
echo "=== All GPU work complete ==="

# Aggregation
echo "Aggregating fine transition scan..."
"$PY" -c "
import sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('$FINE_OUT')
sizes = [300,302,304,306,308,310,312,314,316,318,320]
KEYS = ['max_fitness','mean_fitness','max_size','mean_size','cum_cell_types','cum_pattern_types']

for size in sizes:
    run_dir = ROOT/f'L{size}'/'runs'
    runs = []
    for p in sorted(run_dir.glob('seed_*.csv')):
        arr = np.loadtxt(p, delimiter=',', skiprows=1)
        if arr.ndim == 1: arr = arr[None,:]
        runs.append({
            'step': arr[:,0].astype(int),'max_fitness': arr[:,1],'mean_fitness': arr[:,2],
            'max_size': arr[:,3],'mean_size': arr[:,4],'cum_cell_types': arr[:,5],'cum_pattern_types': arr[:,6]})
    if not runs: print(f'No runs for L={size}'); continue
    steps = runs[0]['step']
    cols = ['step']; summary = {'step': steps}
    for k in KEYS:
        stacked = np.stack([r[k] for r in runs])
        summary[f'{k}_mean'] = stacked.mean(0); summary[f'{k}_std'] = stacked.std(0)
        cols += [f'{k}_mean',f'{k}_std']
    data = np.column_stack([summary[c] for c in cols])
    fmt = ['%d'] + ['%.8f']*(len(cols)-1)
    np.savetxt(ROOT/f'L{size}'/'summary.csv', data, delimiter=',', header=','.join(cols), comments='', fmt=fmt)
    print(f'L={size}: {len(runs)} runs')

sums = []
for size in sizes:
    p = ROOT/f'L{size}'/'summary.csv'
    if not p.exists(): continue
    d = np.genfromtxt(p, delimiter=',', names=True)
    sums.append((size, {n: np.array(d[n]) for n in d.dtype.names}))

if len(sums) >= 2:
    colors = plt.cm.viridis(np.linspace(0.1,0.9,len(sums)))
    for m, yl in [('mean_size_mean','Mean component size'),('max_size_mean','Max component size'),('mean_fitness_mean','Mean fitness')]:
        fig = plt.figure(figsize=(8,5))
        for (sz,s),c in zip(sums,colors):
            plt.plot(s['step'],s[m], label=f'L={sz}', color=c, lw=1.5)
        plt.xlabel('Step'); plt.ylabel(yl); plt.xscale('log'); plt.legend(fontsize=8)
        plt.grid(True,alpha=0.3); plt.title(f'{yl} - Fine Transition Scan'); plt.tight_layout()
        plt.savefig(ROOT/f'{m}_all_sizes.png', dpi=200); plt.close()
    Ls = [s[0] for s in sums]
    for m, yl, logy in [('mean_size_mean','Mean size (final)',True),('max_size_mean','Max size (final)',True),('mean_fitness_mean','Mean fitness (final)',False)]:
        fig = plt.figure(figsize=(8,5))
        vals = [s[1][m][-1] for s in sums]
        plt.plot(Ls, vals, 'o-', lw=2, ms=8); plt.xlabel('L'); plt.ylabel(yl)
        if logy and max(vals)>0: plt.yscale('log')
        plt.grid(True,alpha=0.3); plt.title(f'{yl} vs Grid Size'); plt.tight_layout()
        plt.savefig(ROOT/f'{m}_vs_L.png', dpi=200); plt.close()
    print('Plots saved.')
"

# Combined param sensitivity plots
echo "Combined param sensitivity plots..."
"$PY" scripts/param_sensitivity.py --param mu --values 0.001 0.002 0.005 0.01 \
  --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
  --output-root "$MU_OUT" --progress-interval 50000

"$PY" scripts/param_sensitivity.py --param death_prob --values 0.0005 0.001 0.005 0.01 \
  --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
  --output-root "$DEATH_OUT" --progress-interval 50000

"$PY" scripts/alt_hash_test.py --sizes 200 300 320 400 --runs 5 --steps 10000 \
  --seed0 0 --output-root "$ALT_OUT" --progress-interval 50000

# Statistical analysis
echo "Running statistical analysis..."
"$PY" scripts/statistical_analysis.py \
  --input-root "$ROOT/results/transition_scan" \
  --output-dir "$STATS_OUT"

echo "=== ALL EXPERIMENTS COMPLETE ==="
