#!/usr/bin/env bash
set -euo pipefail

# V3: Pragmatic approach with reduced parameters for feasible runtime
# Key change: leverage 128 CPU cores with many parallel processes

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.02  # ~1GB per process

FINE_OUT="$ROOT/results/fine_transition_scan"
SNAP_OUT="$ROOT/results/snapshots"
MU_OUT="$ROOT/results/param_sensitivity_mu"
DEATH_OUT="$ROOT/results/param_sensitivity_death"
ALT_OUT="$ROOT/results/alt_hash"
STATS_OUT="$ROOT/results/statistical_analysis"

mkdir -p "$FINE_OUT" "$SNAP_OUT" "$MU_OUT" "$DEATH_OUT" "$ALT_OUT" "$STATS_OUT"

# =============================================
# PHASE 1: Fine-grained transition scan
# 11 sizes × 10 seeds = 110 jobs
# 20000 steps, sample every 10 → ~50-100 min per job
# Run 32 in parallel across 4 GPUs
# =============================================
echo "=== PHASE 1: Fine transition scan ==="
echo "Launching 110 fine scan jobs (32 parallel)..."

job_count=0
pids=()
MAX_PARALLEL=32

for size in 300 302 304 306 308 310 312 314 316 318 320; do
  for seed in $(seq 0 9); do
    gpu=$((job_count % 4))
    outfile="$FINE_OUT/L${size}/runs/seed_${seed}.csv"
    if [ -f "$outfile" ]; then
      echo "[skip] $outfile"
      continue
    fi

    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/fast_scan.py \
      --size $size --seed $seed --steps 20000 \
      --sample-interval 10 \
      --output-root "$FINE_OUT" --progress-interval 5000 \
      > "$FINE_OUT/L${size}_s${seed}.log" 2>&1 &
    pids+=($!)
    job_count=$((job_count + 1))

    if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
      # Wait for ANY job to complete (bash wait with no args waits for ALL)
      # Instead, check periodically
      while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 10
      done
      # Clean up finished pids
      new_pids=()
      for pid in "${pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
          new_pids+=($pid)
        fi
      done
      pids=("${new_pids[@]}")
    fi
  done
done

echo "Waiting for all fine scan jobs to complete..."
wait
echo "=== PHASE 1 complete: fine scan ==="

# =============================================
# PHASE 2: Snapshots (2 jobs) + Param sensitivity + Alt hash
# Run snapshots first (they're quick), then param stuff
# =============================================
echo "=== PHASE 2: Snapshots + Param sensitivity + Alt hash ==="

# Snapshots - 2 parallel jobs
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/generate_snapshots.py \
  --sizes 200 --seed 0 --steps 20000 --output-dir "$SNAP_OUT" --progress-interval 2000 \
  > "$SNAP_OUT/L200.log" 2>&1 &
SNAP1=$!

CUDA_VISIBLE_DEVICES=1 "$PY" scripts/generate_snapshots.py \
  --sizes 400 --seed 0 --steps 20000 --output-dir "$SNAP_OUT" --progress-interval 2000 \
  > "$SNAP_OUT/L400.log" 2>&1 &
SNAP2=$!

# Param sensitivity mu: 4 values × 4 sizes × 5 reps = 80 individual runs
# Each run is 10000 steps → ~25-50 min
# Launch all at once distributed across GPUs
echo "Starting param sensitivity mu (80 runs)..."
pids=()
job_count=0

for mu in 0.001 0.002 0.005 0.01; do
  for sz in 200 300 320 400; do
    gpu=$((job_count % 4))
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/param_sensitivity.py \
      --param mu --values $mu --sizes $sz \
      --runs 5 --steps 10000 --seed0 0 \
      --output-root "$MU_OUT" --progress-interval 2000 \
      > "$MU_OUT/mu${mu}_L${sz}.log" 2>&1 &
    pids+=($!)
    job_count=$((job_count + 1))

    if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
      while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 10
      done
      new_pids=()
      for pid in "${pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
          new_pids+=($pid)
        fi
      done
      pids=("${new_pids[@]}")
    fi
  done
done

echo "Starting param sensitivity death (80 runs)..."
for dp in 0.0005 0.001 0.005 0.01; do
  for sz in 200 300 320 400; do
    gpu=$((job_count % 4))
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/param_sensitivity.py \
      --param death_prob --values $dp --sizes $sz \
      --runs 5 --steps 10000 --seed0 0 \
      --output-root "$DEATH_OUT" --progress-interval 2000 \
      > "$DEATH_OUT/death${dp}_L${sz}.log" 2>&1 &
    pids+=($!)
    job_count=$((job_count + 1))

    if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
      while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 10
      done
      new_pids=()
      for pid in "${pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
          new_pids+=($pid)
        fi
      done
      pids=("${new_pids[@]}")
    fi
  done
done

echo "Starting alt hash test..."
for sz in 200 300 320 400; do
  gpu=$((job_count % 4))
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/alt_hash_test.py \
    --sizes $sz --runs 5 --steps 10000 --seed0 0 \
    --output-root "$ALT_OUT" --progress-interval 2000 \
    > "$ALT_OUT/alt_L${sz}.log" 2>&1 &
  pids+=($!)
  job_count=$((job_count + 1))
done

echo "Waiting for all phase 2 jobs..."
wait $SNAP1 $SNAP2
echo "Snapshots complete."
wait
echo "=== PHASE 2 complete ==="

# =============================================
# PHASE 3: Aggregation and analysis
# =============================================
echo "=== PHASE 3: Aggregation and analysis ==="

# Aggregate fine scan
echo "Aggregating fine transition scan..."
"$PY" -c "
import sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('$FINE_OUT')
sizes = [300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320]
KEYS = ['max_fitness', 'mean_fitness', 'max_size', 'mean_size', 'cum_cell_types', 'cum_pattern_types']

for size in sizes:
    run_dir = ROOT / f'L{size}' / 'runs'
    runs = []
    for p in sorted(run_dir.glob('seed_*.csv')):
        arr = np.loadtxt(p, delimiter=',', skiprows=1)
        if arr.ndim == 1: arr = arr[None, :]
        runs.append({'step': arr[:,0].astype(int), 'max_fitness': arr[:,1], 'mean_fitness': arr[:,2],
                      'max_size': arr[:,3], 'mean_size': arr[:,4], 'cum_cell_types': arr[:,5], 'cum_pattern_types': arr[:,6]})
    if not runs:
        print(f'No runs for L={size}'); continue
    steps = runs[0]['step']
    cols = ['step']; summary = {'step': steps}
    for k in KEYS:
        stacked = np.stack([r[k] for r in runs])
        summary[f'{k}_mean'] = stacked.mean(0); summary[f'{k}_std'] = stacked.std(0)
        cols += [f'{k}_mean', f'{k}_std']
    data = np.column_stack([summary[c] for c in cols])
    fmt = ['%d'] + ['%.8f']*(len(cols)-1)
    np.savetxt(ROOT/f'L{size}'/'summary.csv', data, delimiter=',', header=','.join(cols), comments='', fmt=fmt)
    print(f'L={size}: {len(runs)} runs aggregated')

# Plots
sums = []
for size in sizes:
    p = ROOT/f'L{size}'/'summary.csv'
    if not p.exists(): continue
    d = np.genfromtxt(p, delimiter=',', names=True)
    sums.append((size, {n: np.array(d[n]) for n in d.dtype.names}))

if len(sums) >= 2:
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sums)))
    for m, yl in [('mean_size_mean','Mean component size'),('max_size_mean','Max component size'),('mean_fitness_mean','Mean fitness')]:
        fig = plt.figure(figsize=(8,5))
        for (sz, s), c in zip(sums, colors):
            plt.plot(s['step'], s[m], label=f'L={sz}', color=c, lw=1.5)
        plt.xlabel('Step'); plt.ylabel(yl); plt.xscale('log'); plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3); plt.title(f'{yl} - Fine Transition Scan'); plt.tight_layout()
        plt.savefig(ROOT/f'{m}_all_sizes.png', dpi=200); plt.close()

    Ls = [s[0] for s in sums]
    for m, yl, logy in [('mean_size_mean','Mean size (final)',True),('max_size_mean','Max size (final)',True),('mean_fitness_mean','Mean fitness (final)',False)]:
        fig = plt.figure(figsize=(8,5))
        vals = [s[1][m][-1] for s in sums]
        plt.plot(Ls, vals, 'o-', lw=2, ms=8); plt.xlabel('L'); plt.ylabel(yl)
        if logy and max(vals)>0: plt.yscale('log')
        plt.grid(True, alpha=0.3); plt.title(f'{yl} vs Grid Size'); plt.tight_layout()
        plt.savefig(ROOT/f'{m}_vs_L.png', dpi=200); plt.close()
    print('Plots saved.')
"

# Combined param sensitivity plots
echo "Generating combined param sensitivity plots..."
"$PY" scripts/param_sensitivity.py --param mu --values 0.001 0.002 0.005 0.01 \
  --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
  --output-root "$MU_OUT" --progress-interval 50000

"$PY" scripts/param_sensitivity.py --param death_prob --values 0.0005 0.001 0.005 0.01 \
  --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
  --output-root "$DEATH_OUT" --progress-interval 50000

# Alt hash combined plots
"$PY" scripts/alt_hash_test.py --sizes 200 300 320 400 --runs 5 --steps 10000 \
  --seed0 0 --output-root "$ALT_OUT" --progress-interval 50000

# Statistical analysis
echo "Running statistical analysis..."
"$PY" scripts/statistical_analysis.py \
  --input-root "$ROOT/results/transition_scan" \
  --output-dir "$STATS_OUT"

echo "=== ALL EXPERIMENTS COMPLETE ==="
