#!/usr/bin/env bash
set -euo pipefail

# Launch reproduce_figs_4_5_6.py across available GPUs with non-overlapping seed ranges.
# Seeds: 0..86 (total 87 runs, matching the paper's count).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TOTAL_RUNS=87
GPUS=(0 1 2 3)
STEPS=20000
SIZE=100
K=1000
N=10
MU="0.002/0.999"
DEATH=0.001
PROG=200
PY=".venv/bin/python"

CHUNK=$(( (TOTAL_RUNS + ${#GPUS[@]} - 1) / ${#GPUS[@]} ))

for i in "${!GPUS[@]}"; do
  start=$(( i * CHUNK ))
  runs=$(( TOTAL_RUNS - start ))
  (( runs <= 0 )) && continue
  (( runs > CHUNK )) && runs=$CHUNK

  gpu=${GPUS[$i]}
  log="results/figs_gpu${gpu}.log"
  cmd=(uv run --python "$PY" scripts/reproduce_figs_4_5_6.py
       --steps $STEPS --runs $runs --seed0 $start
       --size $SIZE --k $K --n $N --mu "$MU" --death-prob $DEATH
       --progress-interval $PROG)

  printf 'CUDA_VISIBLE_DEVICES=%s %s\n' "$gpu" "${cmd[*]}" > "$log"
  nohup env CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" >> "$log" 2>&1 &
  echo "Launched GPU $gpu seeds ${start}..$((start + runs - 1)) -> $log"
done
