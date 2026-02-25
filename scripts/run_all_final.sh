#!/usr/bin/env bash
set -euo pipefail

# Final orchestration: 4 GPU processes, each runs its workload sequentially
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

FINE_OUT="$ROOT/results/fine_transition_scan"
SNAP_OUT="$ROOT/results/snapshots"
MU_OUT="$ROOT/results/param_sensitivity_mu"
DEATH_OUT="$ROOT/results/param_sensitivity_death"
ALT_OUT="$ROOT/results/alt_hash"
STATS_OUT="$ROOT/results/statistical_analysis"

mkdir -p "$FINE_OUT" "$SNAP_OUT" "$MU_OUT" "$DEATH_OUT" "$ALT_OUT" "$STATS_OUT"

# Each GPU runs a workload function sequentially
# Estimated times per task (L=300-320, sample_interval=10):
#   Fine scan: 6 min/seed, 10 seeds/size = 60 min/size
#   Param sensitivity: 5 reps * ~3 min = 15 min per (param,size) combo
#   Snapshots: ~12 min per size
#   Alt hash: 5 reps * ~3 min = 15 min per size

# GPU 0: Fine scan L=300,302,304 + param mu (L=200,300)
# Estimated: 3*60 + param_mu_200+300 ≈ 3.5h
gpu0_work() {
  export CUDA_VISIBLE_DEVICES=0
  echo "[GPU0] === Fine scan ==="
  for size in 300 302 304; do
    for seed in $(seq 0 9); do
      "$PY" scripts/fast_scan.py --size $size --seed $seed --steps 20000 \
        --sample-interval 10 --output-root "$FINE_OUT" --progress-interval 5000
    done
  done
  echo "[GPU0] === Param mu (L=200,300) ==="
  "$PY" scripts/param_sensitivity.py --param mu --values 0.001 0.002 0.005 0.01 \
    --sizes 200 300 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$MU_OUT" --progress-interval 2000
  echo "[GPU0] DONE"
}

# GPU 1: Fine scan L=306,308,310 + param mu (L=320,400)
gpu1_work() {
  export CUDA_VISIBLE_DEVICES=1
  echo "[GPU1] === Fine scan ==="
  for size in 306 308 310; do
    for seed in $(seq 0 9); do
      "$PY" scripts/fast_scan.py --size $size --seed $seed --steps 20000 \
        --sample-interval 10 --output-root "$FINE_OUT" --progress-interval 5000
    done
  done
  echo "[GPU1] === Param mu (L=320,400) ==="
  "$PY" scripts/param_sensitivity.py --param mu --values 0.001 0.002 0.005 0.01 \
    --sizes 320 400 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$MU_OUT" --progress-interval 2000
  echo "[GPU1] DONE"
}

# GPU 2: Fine scan L=312,316,320 + snapshots L=200 + param death (L=200,300)
gpu2_work() {
  export CUDA_VISIBLE_DEVICES=2
  echo "[GPU2] === Fine scan ==="
  for size in 312 316 320; do
    for seed in $(seq 0 9); do
      "$PY" scripts/fast_scan.py --size $size --seed $seed --steps 20000 \
        --sample-interval 10 --output-root "$FINE_OUT" --progress-interval 5000
    done
  done
  echo "[GPU2] === Snapshots L=200 ==="
  "$PY" scripts/generate_snapshots.py --sizes 200 --seed 0 --steps 20000 \
    --output-dir "$SNAP_OUT" --progress-interval 2000
  echo "[GPU2] === Param death (L=200,300) ==="
  "$PY" scripts/param_sensitivity.py --param death_prob --values 0.0005 0.001 0.005 0.01 \
    --sizes 200 300 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$DEATH_OUT" --progress-interval 2000
  echo "[GPU2] DONE"
}

# GPU 3: Fine scan L=314,318 + snapshots L=400 + param death (L=320,400) + alt hash
gpu3_work() {
  export CUDA_VISIBLE_DEVICES=3
  echo "[GPU3] === Fine scan ==="
  for size in 314 318; do
    for seed in $(seq 0 9); do
      "$PY" scripts/fast_scan.py --size $size --seed $seed --steps 20000 \
        --sample-interval 10 --output-root "$FINE_OUT" --progress-interval 5000
    done
  done
  echo "[GPU3] === Snapshots L=400 ==="
  "$PY" scripts/generate_snapshots.py --sizes 400 --seed 0 --steps 20000 \
    --output-dir "$SNAP_OUT" --progress-interval 2000
  echo "[GPU3] === Param death (L=320,400) ==="
  "$PY" scripts/param_sensitivity.py --param death_prob --values 0.0005 0.001 0.005 0.01 \
    --sizes 320 400 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$DEATH_OUT" --progress-interval 2000
  echo "[GPU3] === Alt hash test ==="
  "$PY" scripts/alt_hash_test.py --sizes 200 300 320 400 --runs 5 --steps 10000 \
    --seed0 0 --output-root "$ALT_OUT" --progress-interval 2000
  echo "[GPU3] DONE"
}

echo "Launching 4 GPU workers..."
gpu0_work > "$ROOT/results/gpu0_final.log" 2>&1 &
GPU0_PID=$!
gpu1_work > "$ROOT/results/gpu1_final.log" 2>&1 &
GPU1_PID=$!
gpu2_work > "$ROOT/results/gpu2_final.log" 2>&1 &
GPU2_PID=$!
gpu3_work > "$ROOT/results/gpu3_final.log" 2>&1 &
GPU3_PID=$!

echo "GPU 0 PID: $GPU0_PID"
echo "GPU 1 PID: $GPU1_PID"
echo "GPU 2 PID: $GPU2_PID"
echo "GPU 3 PID: $GPU3_PID"
echo "Waiting for all GPUs..."

wait $GPU0_PID $GPU1_PID $GPU2_PID $GPU3_PID

echo "=== All GPU work complete ==="

# Post-processing: aggregate fine scan
echo "Aggregating fine transition scan..."
"$PY" -c "
import sys, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('$FINE_OUT')
sizes = [300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320]
METRIC_KEYS = ['max_fitness', 'mean_fitness', 'max_size', 'mean_size', 'cum_cell_types', 'cum_pattern_types']

for size in sizes:
    run_dir = ROOT / f'L{size}' / 'runs'
    runs = []
    for path in sorted(run_dir.glob('seed_*.csv')):
        arr = np.loadtxt(path, delimiter=',', skiprows=1)
        if arr.ndim == 1:
            arr = arr[None, :]
        runs.append({
            'step': arr[:, 0].astype(np.int32),
            'max_fitness': arr[:, 1], 'mean_fitness': arr[:, 2],
            'max_size': arr[:, 3], 'mean_size': arr[:, 4],
            'cum_cell_types': arr[:, 5], 'cum_pattern_types': arr[:, 6],
        })
    if not runs:
        print(f'No runs for L={size}')
        continue
    steps = runs[0]['step']
    summary_cols = ['step']
    summary = {'step': steps}
    for key in METRIC_KEYS:
        stacked = np.stack([r[key] for r in runs])
        summary[f'{key}_mean'] = stacked.mean(axis=0)
        summary[f'{key}_std'] = stacked.std(axis=0)
        summary_cols.extend([f'{key}_mean', f'{key}_std'])
    data = np.column_stack([summary[c] for c in summary_cols])
    fmt = ['%d'] + ['%.8f'] * (len(summary_cols) - 1)
    np.savetxt(ROOT / f'L{size}' / 'summary.csv', data, delimiter=',',
               header=','.join(summary_cols), comments='', fmt=fmt)
    print(f'Aggregated L={size}: {len(runs)} runs')

# Plots
summaries = []
for size in sizes:
    path = ROOT / f'L{size}' / 'summary.csv'
    if not path.exists():
        continue
    data = np.genfromtxt(path, delimiter=',', names=True)
    summaries.append((size, {name: np.array(data[name]) for name in data.dtype.names}))

if len(summaries) >= 2:
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(summaries)))
    for metric, ylabel in [
        ('mean_size_mean', 'Mean component size'),
        ('max_size_mean', 'Max component size'),
        ('mean_fitness_mean', 'Mean fitness'),
    ]:
        plt.figure(figsize=(8, 5))
        for (size, s), c in zip(summaries, colors):
            plt.plot(s['step'], s[metric], label=f'L={size}', color=c, lw=1.5)
        plt.xlabel('Step'); plt.ylabel(ylabel); plt.xscale('log')
        plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
        plt.title(f'{ylabel} - Fine Transition Scan')
        plt.tight_layout()
        plt.savefig(ROOT / f'{metric}_all_sizes.png', dpi=200)
        plt.close()

    L_vals = [s[0] for s in summaries]
    for metric, ylabel, logy in [
        ('mean_size_mean', 'Mean size (final)', True),
        ('max_size_mean', 'Max size (final)', True),
        ('mean_fitness_mean', 'Mean fitness (final)', False),
    ]:
        plt.figure(figsize=(8, 5))
        vals = [s[1][metric][-1] for s in summaries]
        plt.plot(L_vals, vals, 'o-', lw=2, ms=8)
        plt.xlabel('L'); plt.ylabel(ylabel)
        if logy and max(vals) > 0: plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.title(f'{ylabel} vs Grid Size (Fine Scan)')
        plt.tight_layout()
        plt.savefig(ROOT / f'{metric}_vs_L.png', dpi=200)
        plt.close()
    print('Fine scan plots saved.')
"

# Re-run param sensitivity for combined plots
echo "Generating combined param sensitivity plots..."
"$PY" scripts/param_sensitivity.py --param mu --values 0.001 0.002 0.005 0.01 \
  --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
  --output-root "$MU_OUT" --progress-interval 50000

"$PY" scripts/param_sensitivity.py --param death_prob --values 0.0005 0.001 0.005 0.01 \
  --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
  --output-root "$DEATH_OUT" --progress-interval 50000

# Statistical analysis
echo "Running statistical analysis..."
"$PY" scripts/statistical_analysis.py \
  --input-root "$ROOT/results/transition_scan" \
  --output-dir "$STATS_OUT"

echo "=== ALL EXPERIMENTS COMPLETE ==="
