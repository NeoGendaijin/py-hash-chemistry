#!/usr/bin/env bash
set -euo pipefail

# Scan mid-range sizes (L=200..400, step 20) with 10 runs per size.
# Adjust TOTAL_RUNS/STEPS/GPUS as needed for your setup.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TOTAL_RUNS=10
GPUS=(0 1 2 3)
SIZES=(200 220 240 260 280 300 320 340 360 380 400)
STEPS=20000
K=1000
N=10
MU="0.002/0.999"
DEATH=0.001
PROG=500
PY=".venv/bin/python"
OUTROOT="$ROOT/results/transition_scan"

mkdir -p "$OUTROOT"

for size in "${SIZES[@]}"; do
  CHUNK=$(( (TOTAL_RUNS + ${#GPUS[@]} - 1) / ${#GPUS[@]} ))

  for i in "${!GPUS[@]}"; do
    start=$(( i * CHUNK ))
    runs=$(( TOTAL_RUNS - start ))
    (( runs <= 0 )) && continue
    (( runs > CHUNK )) && runs=$CHUNK

    gpu=${GPUS[$i]}
    log="$OUTROOT/L${size}_gpu${gpu}.log"
    cmd=(uv run --python "$PY" scripts/large_space_scaling.py
         --sizes "$size" --steps $STEPS --runs $runs --seed0 $start
         --k $K --n $N --mu "$MU" --death-prob $DEATH
         --progress-interval $PROG --output-root "$OUTROOT"
         --skip-aggregate)

    printf 'CUDA_VISIBLE_DEVICES=%s %s\n' "$gpu" "${cmd[*]}" > "$log"
    nohup env CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" >> "$log" 2>&1 &
    echo "Launched L=${size} GPU $gpu seeds ${start}..$((start + runs - 1)) -> $log"
  done

  wait
  uv run --python "$PY" scripts/large_space_scaling.py \
    --sizes "$size" --steps $STEPS --k $K --n $N --mu "$MU" --death-prob $DEATH \
    --output-root "$OUTROOT" --aggregate-only
done
