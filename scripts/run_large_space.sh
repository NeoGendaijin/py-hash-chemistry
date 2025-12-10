#!/usr/bin/env bash
set -euo pipefail

# Launch large_space_scaling.py across four GPUs for multiple grid sizes.
# Adjust TOTAL_RUNS/STEPS if you want more samples or longer trajectories.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TOTAL_RUNS=100
GPUS=(0 1 2 3)
SIZES=(100 200 400 800)
STEPS=1000000
K=1000
N=10
MU="0.002/0.999"
DEATH=0.001
PROG=200
PY=".venv/bin/python"

mkdir -p results/large_space

for size in "${SIZES[@]}"; do
  CHUNK=$(( (TOTAL_RUNS + ${#GPUS[@]} - 1) / ${#GPUS[@]} ))

  for i in "${!GPUS[@]}"; do
    start=$(( i * CHUNK ))
    runs=$(( TOTAL_RUNS - start ))
    (( runs <= 0 )) && continue
    (( runs > CHUNK )) && runs=$CHUNK

    gpu=${GPUS[$i]}
    log="results/large_space/L${size}_gpu${gpu}.log"
    cmd=(uv run --python "$PY" scripts/large_space_scaling.py
         --sizes "$size" --steps $STEPS --runs $runs --seed0 $start
         --k $K --n $N --mu "$MU" --death-prob $DEATH
         --progress-interval $PROG --output-root "$ROOT/results/large_space"
         --skip-aggregate)

    printf 'CUDA_VISIBLE_DEVICES=%s %s\n' "$gpu" "${cmd[*]}" > "$log"
    nohup env CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" >> "$log" 2>&1 &
    echo "Launched L=${size} GPU $gpu seeds ${start}..$((start + runs - 1)) -> $log"
  done

  wait
  uv run --python "$PY" scripts/large_space_scaling.py \
    --sizes "$size" --steps $STEPS --k $K --n $N --mu "$MU" --death-prob $DEATH \
    --output-root "$ROOT/results/large_space" --aggregate-only
done
