#!/usr/bin/env bash
set -euo pipefail

# Fine-grained transition scan: L=300..320 step=2, 10 replicates, 20000 steps
# Distributes across 4 GPUs

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TOTAL_RUNS=10
GPUS=(0 1 2 3)
SIZES=(300 302 304 306 308 310 312 314 316 318 320)
STEPS=20000
K=1000
N=10
MU="0.002/0.999"
DEATH=0.001
PROG=1000
PY="$ROOT/.venv/bin/python"
OUTROOT="$ROOT/results/fine_transition_scan"

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

    printf 'CUDA_VISIBLE_DEVICES=%s python scripts/large_space_scaling.py L=%s seeds %d..%d\n' "$gpu" "$size" "$start" "$((start + runs - 1))" > "$log"
    nohup env CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/large_space_scaling.py \
         --sizes "$size" --steps $STEPS --runs $runs --seed0 $start \
         --k $K --n $N --mu "$MU" --death-prob $DEATH \
         --progress-interval $PROG --output-root "$OUTROOT" \
         --skip-aggregate >> "$log" 2>&1 &
    echo "Launched L=${size} GPU $gpu seeds ${start}..$((start + runs - 1)) -> $log"
  done

  wait
  "$PY" scripts/large_space_scaling.py \
    --sizes "$size" --steps $STEPS --k $K --n $N --mu "$MU" --death-prob $DEATH \
    --output-root "$OUTROOT" --aggregate-only
done

echo "Fine transition scan complete!"
