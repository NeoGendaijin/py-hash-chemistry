"""ctypes wrapper for the CPU SCHC engine in src/schc_fast.c (built on first use with gcc)."""

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_SRC = Path(__file__).with_name("schc_fast.c")
_LIB = Path(__file__).with_name("_schc_fast.so")
_lib = None

COLUMNS = ["step", "max_fitness", "mean_fitness", "max_size", "mean_size",
           "cum_cell_types", "cum_pattern_types", "n_active", "n_components"]


def _load():
    global _lib
    if _lib is None:
        if not _LIB.exists() or _LIB.stat().st_mtime < _SRC.stat().st_mtime:
            # Parallel scan workers must never load a half-written library.
            with tempfile.TemporaryDirectory(dir=_LIB.parent, prefix=".schc-build-") as tmp:
                target = Path(tmp) / _LIB.name
                subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o",
                                str(target), str(_SRC), "-lm"], check=True)
                os.replace(target, _LIB)
        lib = ctypes.CDLL(str(_LIB))
        i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        lib.schc_run.restype = ctypes.c_int
        lib.schc_run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_double, ctypes.c_void_p, f64p, ctypes.c_int, i32p]
        lib.schc_component_hash.restype = ctypes.c_uint32
        lib.schc_component_hash.argtypes = [i32p, ctypes.c_int, i32p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.schc_forced_contest.restype = None
        lib.schc_forced_contest.argtypes = [i32p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_uint64]
        _lib = lib
    return _lib


SCORES = {"xor": 0, "mix": 1}
RULES = {"legacy": 0, "original": 1}


def run(L, seed, steps, k=1000, n=10, mu=0.002 / 0.999, death_prob=0.001, score="mix", winner_rule="original",
        sample_interval=10, stop_mean_size=0.0, init_config=None):
    """Run one simulation; returns (rows as float64 array with COLUMNS, final configuration)."""
    if L < 1 or k < 1 or not 0 <= n <= L * L or steps < 0 or sample_interval < 1:
        raise ValueError("invalid grid size, cell count, steps or sampling interval")
    if not 0 <= mu <= 1 or not 0 <= death_prob <= 1 or seed < 0:
        raise ValueError("probabilities must be in [0, 1] and seed must be non-negative")
    if init_config is not None:
        initial = np.asarray(init_config)
        if initial.shape != (L, L) or not np.all((initial >= 0) & (initial <= k)):
            raise ValueError("initial configuration must have shape (L, L) and cell types 0..k")
    lib = _load()
    max_rows = steps // sample_interval + 2
    rows = np.zeros((max_rows, len(COLUMNS)), dtype=np.float64)
    final = np.zeros((L, L), dtype=np.int32)
    init = None if init_config is None else np.ascontiguousarray(init_config, dtype=np.int32)
    nrows = lib.schc_run(L, n, k, mu, death_prob, SCORES[score], RULES[winner_rule], seed, steps,
                         sample_interval, stop_mean_size,
                         None if init is None else init.ctypes.data_as(ctypes.c_void_p), rows, max_rows, final)
    return rows[:nrows], final


def component_hash(config, cells, score="mix", seed=0):
    config = np.ascontiguousarray(config, dtype=np.int32)
    cells = np.ascontiguousarray(cells, dtype=np.int32)
    return int(_load().schc_component_hash(config.reshape(-1), config.shape[0], cells, len(cells), SCORES[score], seed))


def forced_contest(config, start1, start2, coin, k=1000, mu=0.0, death_prob=0.0, score="mix",
                   winner_rule="original", seed=0):
    """One contest from the given start cells (flat indices) and tie coin; returns the new configuration."""
    out = np.ascontiguousarray(config, dtype=np.int32).copy()
    _load().schc_forced_contest(out.reshape(-1), out.shape[0], k, mu, death_prob, SCORES[score],
                                RULES[winner_rule], start1, start2, coin, seed)
    return out
