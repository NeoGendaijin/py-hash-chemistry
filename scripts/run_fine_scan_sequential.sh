#!/usr/bin/env bash
set -euo pipefail

# Fine-grained transition scan: one process per GPU, running sequentially per GPU
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.05

FINE_OUT="$ROOT/results/fine_transition_scan"
mkdir -p "$FINE_OUT"

# Distribute 11 sizes x 10 seeds across 4 GPUs
# GPU 0: L=300 (10 seeds), L=304 (10 seeds), L=308 (10 seeds)
# GPU 1: L=302 (10 seeds), L=306 (10 seeds), L=310 (10 seeds)
# GPU 2: L=312 (10 seeds), L=316 (10 seeds), L=320 (10 seeds)
# GPU 3: L=314 (10 seeds), L=318 (10 seeds)

run_gpu() {
  local gpu=$1
  shift
  local sizes=("$@")

  export CUDA_VISIBLE_DEVICES=$gpu
  for size in "${sizes[@]}"; do
    for seed in $(seq 0 9); do
      outfile="$FINE_OUT/L${size}/runs/seed_${seed}.csv"
      if [ -f "$outfile" ]; then
        echo "[GPU$gpu skip] $outfile exists"
        continue
      fi
      echo "[GPU$gpu] L=$size seed=$seed starting..."
      "$PY" scripts/fast_scan.py \
        --size "$size" --seed $seed --steps 20000 \
        --sample-interval 10 \
        --output-root "$FINE_OUT" --progress-interval 5000
    done
  done
  echo "[GPU$gpu] All done"
}

run_gpu 0 300 304 308 > "$FINE_OUT/gpu0_seq.log" 2>&1 &
run_gpu 1 302 306 310 > "$FINE_OUT/gpu1_seq.log" 2>&1 &
run_gpu 2 312 316 320 > "$FINE_OUT/gpu2_seq.log" 2>&1 &
run_gpu 3 314 318     > "$FINE_OUT/gpu3_seq.log" 2>&1 &

echo "Fine scan launched on 4 GPUs (sequential per GPU)"
wait
echo "Fine scan complete!"
