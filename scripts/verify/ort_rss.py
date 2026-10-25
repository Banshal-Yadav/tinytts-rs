"""A3: ORT peak RSS on the 512-token input, 1 and 4 threads.

Reads the child process's peak working set via GetProcessMemoryInfo on a handle
opened with PROCESS_QUERY_INFORMATION, which is what the earlier attempt got
wrong (it queried the Python process, not the ORT worker).

Usage:  python scripts/verify/ort_rss.py
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import statistics
import subprocess
import sys
import textwrap
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _ids
MODEL = os.path.join(ROOT, "paradee_int8.onnx")
IDS = _ids.all_ids()

WORKER = os.path.join(ROOT, "scripts", "_ort_worker.py")

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
        ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t),
        ("g", ctypes.c_size_t),
    ]


def peak_of_pid(pid):
    PROCESS_QUERY_INFORMATION = 0x0400
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
    if not h:
        return None
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    ok = k32.K32GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
    k32.CloseHandle(h)
    if not ok:
        return None
    return c.PeakWorkingSetSize / 1e6


def main():
    ids = IDS["long"]
    print(f"input: long, {len(ids)} tokens")
    for threads in (1, 4):
        p = subprocess.Popen(
            [sys.executable, WORKER, str(threads), ",".join(str(i) for i in ids)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        peak = 0.0
        while p.poll() is None:
            v = peak_of_pid(p.pid)
            if v and v > peak:
                peak = v
            time.sleep(0.02)
        out, err = p.communicate()
        print(f"ORT threads={threads}  peak RSS = {peak:7.1f} MB")
        for line in out.strip().splitlines():
            print("   ", line)
        if err.strip():
            print("    stderr:", err.strip()[:200])


if __name__ == "__main__":
    main()
