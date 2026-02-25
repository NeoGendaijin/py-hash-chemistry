#!/usr/bin/env bash
set -euo pipefail

# All experiments distributed across 4 GPUs with memory-efficient JAX settings
# Allows multiple processes per GPU since JAX won't preallocate all GPU memory

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.05

FINE_OUT="$ROOT/results/fine_transition_scan"
SNAP_OUT="$ROOT/results/snapshots"
MU_OUT="$ROOT/results/param_sensitivity_mu"
DEATH_OUT="$ROOT/results/param_sensitivity_death"
ALT_OUT="$ROOT/results/alt_hash"

mkdir -p "$FINE_OUT" "$SNAP_OUT" "$MU_OUT" "$DEATH_OUT" "$ALT_OUT"

MAX_PARALLEL=10  # processes per GPU (CPU-bound, 128 cores available)

# =========================================================
# PHASE 1: Fine-grained transition scan
# 11 sizes x 10 seeds = 110 runs, distributed across 4 GPUs
# =========================================================
echo "=== PHASE 1: Fine transition scan ==="
ALL_SIZES=(300 302 304 306 308 310 312 314 316 318 320)
STEPS=20000

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
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/large_space_scaling.py \
      --sizes "$size" --steps $STEPS --runs 1 --seed0 $seed \
      --k 1000 --n 10 --mu "0.002/0.999" --death-prob 0.001 \
      --progress-interval 5000 --output-root "$FINE_OUT" \
      --skip-aggregate > "$log" 2>&1 &
    pids+=($!)
    gpu_idx=$((gpu_idx + 1))

    # Limit parallelism
    if [ ${#pids[@]} -ge $((MAX_PARALLEL * 4)) ]; then
      # Wait for any to finish
      for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
      done
      pids=()
    fi
  done
done

# Wait for all fine scan jobs
echo "Waiting for fine scan to complete..."
for pid in "${pids[@]}"; do
  wait $pid 2>/dev/null || true
done
pids=()

echo "Aggregating fine scan..."
"$PY" scripts/large_space_scaling.py \
  --sizes ${ALL_SIZES[@]} --steps $STEPS \
  --k 1000 --n 10 --mu "0.002/0.999" --death-prob 0.001 \
  --output-root "$FINE_OUT" --aggregate-only

echo "=== PHASE 1 complete ==="

# =========================================================
# PHASE 2: Snapshots + Param sensitivity + Alt hash
# Run in parallel across GPUs
# =========================================================
echo "=== PHASE 2: Snapshots + Param sensitivity + Alt hash ==="

# Snapshots on GPU 0 and 1
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/generate_snapshots.py \
  --sizes 200 --seed 0 --steps 20000 --output-dir "$SNAP_OUT" \
  --progress-interval 2000 > "$SNAP_OUT/L200.log" 2>&1 &
pids+=($!)

CUDA_VISIBLE_DEVICES=1 "$PY" scripts/generate_snapshots.py \
  --sizes 400 --seed 0 --steps 20000 --output-dir "$SNAP_OUT" \
  --progress-interval 2000 > "$SNAP_OUT/L400.log" 2>&1 &
pids+=($!)

# Param sensitivity mu - run each (param_val, size) combo separately
echo "Starting param sensitivity mu..."
MU_VALS=(0.001 0.002 0.005 0.01)
PARAM_SIZES=(200 300 320 400)

for mu in "${MU_VALS[@]}"; do
  for sz in "${PARAM_SIZES[@]}"; do
    gpu=$((gpu_idx % 4))
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/param_sensitivity.py \
      --param mu --values $mu --sizes $sz \
      --runs 5 --steps 10000 --seed0 0 \
      --output-root "$MU_OUT" --progress-interval 2000 \
      > "$MU_OUT/mu${mu}_L${sz}.log" 2>&1 &
    pids+=($!)
    gpu_idx=$((gpu_idx + 1))

    if [ ${#pids[@]} -ge $((MAX_PARALLEL * 4)) ]; then
      for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
      done
      pids=()
    fi
  done
done

# Param sensitivity death
echo "Starting param sensitivity death..."
DEATH_VALS=(0.0005 0.001 0.005 0.01)

for dp in "${DEATH_VALS[@]}"; do
  for sz in "${PARAM_SIZES[@]}"; do
    gpu=$((gpu_idx % 4))
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/param_sensitivity.py \
      --param death_prob --values $dp --sizes $sz \
      --runs 5 --steps 10000 --seed0 0 \
      --output-root "$DEATH_OUT" --progress-interval 2000 \
      > "$DEATH_OUT/death${dp}_L${sz}.log" 2>&1 &
    pids+=($!)
    gpu_idx=$((gpu_idx + 1))

    if [ ${#pids[@]} -ge $((MAX_PARALLEL * 4)) ]; then
      for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
      done
      pids=()
    fi
  done
done

# Alt hash test
echo "Starting alt hash test..."
for sz in "${PARAM_SIZES[@]}"; do
  gpu=$((gpu_idx % 4))
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/alt_hash_test.py \
    --sizes $sz --runs 5 --steps 10000 --seed0 0 \
    --output-root "$ALT_OUT" --progress-interval 2000 \
    > "$ALT_OUT/alt_L${sz}.log" 2>&1 &
  pids+=($!)
  gpu_idx=$((gpu_idx + 1))
done

echo "Waiting for phase 2..."
for pid in "${pids[@]}"; do
  wait $pid 2>/dev/null || true
done

echo "=== PHASE 2 complete ==="

# =========================================================
# PHASE 3: Aggregation and plots for param sensitivity
# =========================================================
echo "=== PHASE 3: Final aggregation ==="

# Re-run param sensitivity to generate combined plots
"$PY" scripts/param_sensitivity.py \
  --param mu --values ${MU_VALS[@]} \
  --sizes ${PARAM_SIZES[@]} --runs 5 --steps 10000 --seed0 0 \
  --output-root "$MU_OUT" --progress-interval 10000

"$PY" scripts/param_sensitivity.py \
  --param death_prob --values ${DEATH_VALS[@]} \
  --sizes ${PARAM_SIZES[@]} --runs 5 --steps 10000 --seed0 0 \
  --output-root "$DEATH_OUT" --progress-interval 10000

# Re-run alt hash to generate combined plots
"$PY" scripts/alt_hash_test.py \
  --sizes ${PARAM_SIZES[@]} --runs 5 --steps 10000 --seed0 0 \
  --output-root "$ALT_OUT" --progress-interval 10000

# =========================================================
# PHASE 4: Statistical analysis
# =========================================================
echo "=== PHASE 4: Statistical analysis ==="
"$PY" scripts/statistical_analysis.py \
  --input-root "$ROOT/results/transition_scan" \
  --output-dir "$ROOT/results/statistical_analysis"

echo "=== ALL EXPERIMENTS COMPLETE ==="
