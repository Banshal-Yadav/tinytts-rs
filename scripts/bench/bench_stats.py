"""p50/p95 over >= 20 runs after 3 warmups, for short/mid/long.

Usage:  python scripts/bench/bench_stats.py [runs]
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _ids
EXE = os.environ.get("CARGO_TARGET_DIR", os.path.join(ROOT, "runtime", "target")) + r"\release\tinytts.exe"
MODEL = os.path.join(ROOT, "paradee_int8.onnx")
IDS = _ids.all_ids()

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
WARM = 3

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
        ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t),
        ("g", ctypes.c_size_t),
    ]


def peak_mb():
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 1e6


def run_once(ids):
    p = subprocess.run([EXE, MODEL, "--profile"] + [str(i) for i in ids],
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    synth = None
    for line in out.splitlines():
        if "synth" in line and ":" in line:
            synth = float(line.split(":")[1].strip().split()[0])
    return synth


def main():
    print(f"runs={RUNS} warmup={WARM}")
    for name in ("short", "mid", "long"):
        ids = IDS[name]
        for _ in range(WARM):
            run_once(ids)
        ts = []
        for _ in range(RUNS):
            t = run_once(ids)
            if t is not None:
                ts.append(t)
        ts.sort()
        p50 = statistics.median(ts)
        p95 = ts[int(0.95 * (len(ts) - 1))]
        print(f"{name:6} tokens={len(ids):4}  n={len(ts):3}  "
              f"min={ts[0]:8.1f}  p50={p50:8.1f}  p95={p95:8.1f}  max={ts[-1]:8.1f} ms")


if __name__ == "__main__":
    main()
