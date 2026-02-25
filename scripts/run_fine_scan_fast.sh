#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.05

FINE_OUT="$ROOT/results/fine_transition_scan"
mkdir -p "$FINE_OUT"

ALL_SIZES=(300 302 304 306 308 310 312 314 316 318 320)
STEPS=20000
SAMPLE=10  # sample metrics every 10 steps -> 10x speedup

MAX_JOBS=40  # 10 per GPU
running=0
pids=()
gpu_idx=0

for size in "${ALL_SIZES[@]}"; do
  for seed in $(seq 0 9); do
    gpu=$((gpu_idx % 4))
    outfile="$FINE_OUT/L${size}/runs/seed_${seed}.csv"
    if [ -f "$outfile" ]; then
      echo "[skip] $outfile exists"
      continue
    fi

    log="$FINE_OUT/L${size}_seed${seed}.log"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/fast_scan.py \
      --size "$size" --seed $seed --steps $STEPS \
      --sample-interval $SAMPLE \
      --output-root "$FINE_OUT" --progress-interval 2000 \
      > "$log" 2>&1 &
    pids+=($!)
    gpu_idx=$((gpu_idx + 1))
    running=$((running + 1))

    # Limit parallelism
    if [ $running -ge $MAX_JOBS ]; then
      for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
      done
      pids=()
      running=0
      echo "Batch complete. Starting next batch..."
    fi
  done
done

# Wait for remaining
echo "Waiting for final batch..."
for pid in "${pids[@]}"; do
  wait $pid 2>/dev/null || true
done

echo "All fine scan runs complete!"

# Aggregate: need a custom aggregation since step counts differ from original
echo "Aggregating..."
"$PY" -c "
import sys, numpy as np
from pathlib import Path

ROOT = Path('$FINE_OUT')
sizes = [${ALL_SIZES[@]/%/,}]
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
            'max_fitness': arr[:, 1],
            'mean_fitness': arr[:, 2],
            'max_size': arr[:, 3],
            'mean_size': arr[:, 4],
            'cum_cell_types': arr[:, 5],
            'cum_pattern_types': arr[:, 6],
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
    outpath = ROOT / f'L{size}' / 'summary.csv'
    np.savetxt(outpath, data, delimiter=',', header=','.join(summary_cols), comments='', fmt=fmt)
    print(f'Aggregated L={size}: {len(runs)} runs, {len(steps)} samples each')

# Multi-size comparison plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
            plt.plot(s['step'], s[metric], label=f'L={size}', color=c, linewidth=1.5)
        plt.xlabel('Step')
        plt.ylabel(ylabel)
        plt.xscale('log')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.title(f'{ylabel} - Fine Transition Scan')
        plt.tight_layout()
        plt.savefig(ROOT / f'{metric}_all_sizes.png', dpi=200)
        plt.close()

    # Final-value vs L
    L_vals = [s[0] for s in summaries]
    for metric, ylabel, logy in [
        ('mean_size_mean', 'Mean size (final)', True),
        ('max_size_mean', 'Max size (final)', True),
        ('mean_fitness_mean', 'Mean fitness (final)', False),
    ]:
        plt.figure(figsize=(8, 5))
        vals = [s[1][metric][-1] for s in summaries]
        plt.plot(L_vals, vals, 'o-', linewidth=2, markersize=8)
        plt.xlabel('L (grid size)')
        plt.ylabel(ylabel)
        if logy and max(vals) > 0:
            plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.title(f'{ylabel} vs Grid Size')
        plt.tight_layout()
        plt.savefig(ROOT / f'{metric}_vs_L.png', dpi=200)
        plt.close()

    print('Plots saved.')
"

echo "Fine transition scan fully complete!"
