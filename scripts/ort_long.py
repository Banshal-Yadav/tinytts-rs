"""ORT on the 512-token input, 1 and 4 threads: time and peak RSS.

Usage:  python scripts/ort_long.py
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import _ids
MODEL = os.path.join(ROOT, "paradee_int8.onnx")
IDS = _ids.all_ids()

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
        ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t),
        ("g", ctypes.c_size_t),
    ]


def rss():
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1e6, c.PeakWorkingSetSize / 1e6


def bench(threads, ids, runs=20, warm=3):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    t0 = time.perf_counter()
    s = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    load = (time.perf_counter() - t0) * 1e3
    a = np.array([ids], dtype=np.int64)
    sp = np.array([1.0], dtype=np.float32)
    for _ in range(warm):
        s.run(None, {"input_ids": a, "speed": sp})
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = s.run(None, {"input_ids": a, "speed": sp})
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    cur, pk = rss()
    audio = out[0].shape[-1] / 24000
    p50 = statistics.median(ts)
    p95 = ts[int(0.95 * (len(ts) - 1))]
    print(f"ORT threads={threads}  load={load:7.1f} ms  "
          f"min={ts[0]:8.1f}  p50={p50:8.1f}  p95={p95:8.1f} ms  "
          f"audio={audio:6.3f}s  RTF={audio/(p50/1e3):6.2f}x  "
          f"rss={cur:6.1f} peak={pk:6.1f} MB")


def main():
    ids = IDS["long"]
    print(f"input: long, {len(ids)} tokens")
    for th in (1, 4):
        bench(th, ids)


if __name__ == "__main__":
    main()
