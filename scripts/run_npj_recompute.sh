#!/usr/bin/env bash
# CPU re-computation of npj Sec. 4 with the corrected model (winner_rule=original, score=mix),
# using the validated CPU engine. Run from the repository root, next to
#   uv run --no-sync python scripts/thermal_watchdog.py --pattern run_scan_cpu.py --grace 20 \
#       --stop-file results/npj_recompute.STOP --log results/npj_recompute_thermal.csv
# Jobs (see report/20261022_npj再計算リスト.md):
#   J1  size scan without early stop: coarse L=200..400 (step 20) + onset region L=100..250, 100 seeds
#       (runs of results/cpu_scan/original_mix that never stopped are copied: identical by determinism)
#   J2  long-run novelty at L=200, metrics every step, 40 seeds
#   J3  configurations for the snapshot panels (L=200 seed 0 at 5k/10k; L=400 seed 5 at 2k/5k/10k)
#   J4  mutation / death-probability sensitivity, 10k steps, 30 seeds, no early stop
#   J5  40k-step horizon check (L=240/320/400, 100 seeds, early stop at mean size 2000)
set -u
cd "$(dirname "$0")/.."
PY="uv run --no-sync python"
W=24
STOP=results/npj_recompute.STOP
run() {
    if [ -e "$STOP" ]; then echo "stop file $STOP present (thermal watchdog); aborting"; exit 1; fi
    echo "== $(date '+%F %T') $*"
    "$@"
}

# J1 -------------------------------------------------------------------------------------------
FULL=results/cpu_scan_full
run $PY - <<'EOF'
import shutil
from pathlib import Path
import pandas as pd
src = {L: Path(f"results/cpu_scan/original_mix/L{L}") for L in range(200, 401, 20)}
src[100] = Path("results/cpu_scan_check/original_mix/L100")
n = 0
for L, d in src.items():
    for s in range(100):
        f = d / "runs" / f"seed_{s}.csv"
        if not f.exists() or pd.read_csv(f).mean_size.iloc[-1] >= 2000:
            continue          # stopped at the runaway threshold: must be re-run without stop
        out = Path(f"results/cpu_scan_full/original_mix/L{L}")
        (out / "runs").mkdir(parents=True, exist_ok=True); (out / "configs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / "runs" / f.name)
        cfg = d / "configs" / f"seed_{s}_final.npy"
        if cfg.exists():
            shutil.copy2(cfg, out / "configs" / cfg.name)
        n += 1
print(f"copied {n} runs that never reached the early-stop threshold")
EOF
run $PY scripts/run_scan_cpu.py --sizes 100 120 140 160 180 200 210 220 230 240 250 260 280 300 320 340 360 380 400 \
    --seeds 0-99 --stop-mean-size 0 --workers $W --output-root $FULL

# J2 -------------------------------------------------------------------------------------------
run $PY scripts/run_scan_cpu.py --sizes 200 --seeds 0-39 --sample-interval 1 --stop-mean-size 0 --workers $W \
    --output-root results/cpu_novelty

# J3 -------------------------------------------------------------------------------------------
for T in 5000 10000; do
    run $PY scripts/run_scan_cpu.py --sizes 200 --seeds 0 --steps $T --stop-mean-size 0 --workers 1 \
        --output-root results/cpu_snapshots/t$T
done
for T in 2000 5000 10000; do
    run $PY scripts/run_scan_cpu.py --sizes 400 --seeds 5 --steps $T --stop-mean-size 0 --workers 1 \
        --output-root results/cpu_snapshots/t$T
done

# J4 -------------------------------------------------------------------------------------------
SENS="--sizes 100 200 240 300 320 400 --seeds 0-29 --steps 10000 --stop-mean-size 0 --workers $W --output-root results/cpu_sensitivity"
for MU in 0.001 0.002 0.005 0.01; do run $PY scripts/run_scan_cpu.py $SENS --mu $MU; done
for PD in 0.0005 0.001 0.005 0.01; do run $PY scripts/run_scan_cpu.py $SENS --death-prob $PD; done

# J5 -------------------------------------------------------------------------------------------
run $PY scripts/run_scan_cpu.py --sizes 240 320 400 --seeds 0-99 --steps 40000 --workers $W --output-root results/cpu_scan_40k

echo "== $(date '+%F %T') all jobs done"
