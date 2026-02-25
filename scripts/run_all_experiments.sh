#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
FINE_OUT="$ROOT/results/fine_transition_scan"
SNAP_OUT="$ROOT/results/snapshots"
MU_OUT="$ROOT/results/param_sensitivity_mu"
DEATH_OUT="$ROOT/results/param_sensitivity_death"
ALT_OUT="$ROOT/results/alt_hash"

mkdir -p "$FINE_OUT" "$SNAP_OUT" "$MU_OUT" "$DEATH_OUT" "$ALT_OUT"

# ============================================================
# GPU 0: Fine scan L=300,302,304,306 (10 seeds each)
#         then param mu values 0.001, 0.002 (all 4 sizes)
# ============================================================
(
  export CUDA_VISIBLE_DEVICES=0
  echo "[GPU0] Starting fine scan L=300,302,304,306"
  for size in 300 302 304 306; do
    "$PY" scripts/large_space_scaling.py \
      --sizes "$size" --steps 20000 --runs 10 --seed0 0 \
      --k 1000 --n 10 --mu "0.002/0.999" --death-prob 0.001 \
      --progress-interval 2000 --output-root "$FINE_OUT"
  done
  echo "[GPU0] Fine scan done. Starting param sensitivity mu=0.001,0.002"
  "$PY" scripts/param_sensitivity.py \
    --param mu --values 0.001 0.002 \
    --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$MU_OUT" --progress-interval 1000
  echo "[GPU0] All done"
) > "$ROOT/results/gpu0.log" 2>&1 &
echo "Launched GPU 0"

# ============================================================
# GPU 1: Fine scan L=308,310,312 (10 seeds each)
#         then param mu values 0.005, 0.01 (all 4 sizes)
# ============================================================
(
  export CUDA_VISIBLE_DEVICES=1
  echo "[GPU1] Starting fine scan L=308,310,312"
  for size in 308 310 312; do
    "$PY" scripts/large_space_scaling.py \
      --sizes "$size" --steps 20000 --runs 10 --seed0 0 \
      --k 1000 --n 10 --mu "0.002/0.999" --death-prob 0.001 \
      --progress-interval 2000 --output-root "$FINE_OUT"
  done
  echo "[GPU1] Fine scan done. Starting param sensitivity mu=0.005,0.01"
  "$PY" scripts/param_sensitivity.py \
    --param mu --values 0.005 0.01 \
    --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$MU_OUT" --progress-interval 1000
  echo "[GPU1] All done"
) > "$ROOT/results/gpu1.log" 2>&1 &
echo "Launched GPU 1"

# ============================================================
# GPU 2: Fine scan L=314,316 + snapshots L=200
#         then param death values 0.0005, 0.001 (all 4 sizes)
# ============================================================
(
  export CUDA_VISIBLE_DEVICES=2
  echo "[GPU2] Starting fine scan L=314,316"
  for size in 314 316; do
    "$PY" scripts/large_space_scaling.py \
      --sizes "$size" --steps 20000 --runs 10 --seed0 0 \
      --k 1000 --n 10 --mu "0.002/0.999" --death-prob 0.001 \
      --progress-interval 2000 --output-root "$FINE_OUT"
  done
  echo "[GPU2] Fine scan done. Starting snapshots L=200"
  "$PY" scripts/generate_snapshots.py \
    --sizes 200 --seed 0 --steps 20000 --output-dir "$SNAP_OUT" --progress-interval 1000
  echo "[GPU2] Snapshots L=200 done. Starting param sensitivity death=0.0005,0.001"
  "$PY" scripts/param_sensitivity.py \
    --param death_prob --values 0.0005 0.001 \
    --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$DEATH_OUT" --progress-interval 1000
  echo "[GPU2] All done"
) > "$ROOT/results/gpu2.log" 2>&1 &
echo "Launched GPU 2"

# ============================================================
# GPU 3: Fine scan L=318,320 + snapshots L=400
#         then param death values 0.005, 0.01 (all 4 sizes)
#         then alt hash test
# ============================================================
(
  export CUDA_VISIBLE_DEVICES=3
  echo "[GPU3] Starting fine scan L=318,320"
  for size in 318 320; do
    "$PY" scripts/large_space_scaling.py \
      --sizes "$size" --steps 20000 --runs 10 --seed0 0 \
      --k 1000 --n 10 --mu "0.002/0.999" --death-prob 0.001 \
      --progress-interval 2000 --output-root "$FINE_OUT"
  done
  echo "[GPU3] Fine scan done. Starting snapshots L=400"
  "$PY" scripts/generate_snapshots.py \
    --sizes 400 --seed 0 --steps 20000 --output-dir "$SNAP_OUT" --progress-interval 1000
  echo "[GPU3] Snapshots L=400 done. Starting param sensitivity death=0.005,0.01"
  "$PY" scripts/param_sensitivity.py \
    --param death_prob --values 0.005 0.01 \
    --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$DEATH_OUT" --progress-interval 1000
  echo "[GPU3] Param death done. Starting alt hash test"
  "$PY" scripts/alt_hash_test.py \
    --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
    --output-root "$ALT_OUT" --progress-interval 1000
  echo "[GPU3] All done"
) > "$ROOT/results/gpu3.log" 2>&1 &
echo "Launched GPU 3"

echo "All 4 GPUs launched. Waiting for completion..."
wait
echo "All GPU tasks complete!"

# Post-processing: aggregate fine transition scan multi-size plots
echo "Generating multi-size plots for fine transition scan..."
"$PY" scripts/large_space_scaling.py \
  --sizes 300 302 304 306 308 310 312 314 316 318 320 \
  --steps 20000 --output-root "$FINE_OUT" --aggregate-only

# Generate combined param sensitivity plots
echo "Generating combined param sensitivity plots..."
"$PY" scripts/param_sensitivity.py \
  --param mu --values 0.001 0.002 0.005 0.01 \
  --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
  --output-root "$MU_OUT" --progress-interval 1000

"$PY" scripts/param_sensitivity.py \
  --param death_prob --values 0.0005 0.001 0.005 0.01 \
  --sizes 200 300 320 400 --runs 5 --steps 10000 --seed0 0 \
  --output-root "$DEATH_OUT" --progress-interval 1000

# Statistical analysis (uses existing transition_scan data + fine scan)
echo "Running statistical analysis..."
"$PY" scripts/statistical_analysis.py \
  --input-root "$ROOT/results/transition_scan" \
  --output-dir "$ROOT/results/statistical_analysis"

echo "ALL EXPERIMENTS COMPLETE!"
