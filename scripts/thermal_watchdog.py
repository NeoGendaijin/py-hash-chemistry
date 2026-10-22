#!/usr/bin/env python3
"""
Thermal watchdog for long simulation runs.

Polls the CPU package temperatures (k10temp Tctl, every socket) and the GPU
temperatures every --interval seconds and appends them to a CSV log. If any CPU
Tctl exceeds --cpu-max or any GPU exceeds --gpu-max for --patience consecutive
polls, it sends SIGTERM to every process whose command line contains --pattern
(e.g. the runner script name), records that in the log, and exits.

Motivation: an earlier L=400 scan on this machine overheated the CPU and took
the server down.

Example:
  python scripts/thermal_watchdog.py --pattern run_spatial_dyadic_sweep.py \
      --log results/thermal.csv --cpu-max 80 --gpu-max 85
"""

from __future__ import annotations

import argparse
import glob
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


def cpu_tctl() -> list[float]:
    temps = []
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if Path(hw, "name").read_text().strip() != "k10temp":
                continue
            for label in glob.glob(f"{hw}/temp*_label"):
                if Path(label).read_text().strip() == "Tctl":
                    temps.append(int(Path(label.replace("_label", "_input")).read_text()) / 1000.0)
        except OSError:
            continue
    return temps


def gpu_temps() -> list[float]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return [float(line.split(",")[0]) for line in out.strip().splitlines() if line.strip()]
    except Exception:
        return []


def matching_pids(pattern: str) -> list[int]:
    pids = []
    for d in glob.glob("/proc/[0-9]*"):
        pid = int(os.path.basename(d))
        if pid == os.getpid():
            continue
        try:
            cmd = Path(d, "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if pattern in cmd and "thermal_watchdog" not in cmd:
            pids.append(pid)
    return pids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", required=True, help="substring of the command line of processes to stop")
    ap.add_argument("--log", required=True)
    ap.add_argument("--cpu-max", type=float, default=80.0)
    ap.add_argument("--gpu-max", type=float, default=85.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--grace", type=int, default=0,
                    help="exit only after this many consecutive polls without matching processes (for job chains)")
    ap.add_argument("--stop-file", default=None, help="touched on an overheat stop, so that job chains can abort")
    args = ap.parse_args()

    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    if not log.exists():
        log.write_text("time,cpu_tctl_max,gpu_max,cpu_all,gpu_all,event\n")
    strikes = 0
    idle = 0
    while True:
        cpu, gpu = cpu_tctl(), gpu_temps()
        cmax, gmax = max(cpu, default=float("nan")), max(gpu, default=float("nan"))
        hot = (cpu and cmax > args.cpu_max) or (gpu and gmax > args.gpu_max)
        strikes = strikes + 1 if hot else 0
        event = ""
        pids = matching_pids(args.pattern)
        if strikes >= args.patience:
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
            event = f"STOPPED pids {pids} (cpu {cmax} > {args.cpu_max} or gpu {gmax} > {args.gpu_max})"
            if args.stop_file:
                Path(args.stop_file).touch()
        with log.open("a") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')},{cmax},{gmax},"
                    f"{' '.join(map(str, cpu))},{' '.join(map(str, gpu))},{event}\n")
        idle = 0 if pids else idle + 1
        if event or idle > args.grace:
            if not pids:
                with log.open("a") as f:
                    f.write(f"{datetime.now().isoformat(timespec='seconds')},,,,,no matching processes left; exiting\n")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
