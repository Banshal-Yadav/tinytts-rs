"""Benchmark: Python/ONNX-Runtime baseline vs our Rust runtime.

Measures, for the same 52-token phoneme input:
  * load time (model parse + weight setup)
  * synthesis time (first call, and steady-state median over N runs)
  * peak working set (RSS) while loaded
  * real-time factor (audio seconds produced per wall second)

Usage:
    python scripts/benchmark.py            # both
    python scripts/benchmark.py python     # baseline only
    python scripts/benchmark.py rust       # ours only
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "paradee_int8.onnx")
RUST_EXE = os.path.join(ROOT, "runtime", "target", "release", "tinytts.exe")
SAMPLE_RATE = 24000

# 52-token "Hello, this is a test of the speech synthesis system."
IDS = [
    0, 50, 86, 54, 57, 135, 3, 16, 81, 102, 61, 16, 102, 68, 16, 83, 16, 62,
    86, 61, 62, 16, 71, 64, 16, 81, 83, 16, 61, 58, 51, 158, 62, 131, 16, 61,
    102, 56, 119, 83, 61, 102, 61, 16, 61, 102, 61, 62, 83, 55, 4, 0,
]

RUNS = 5

# --- Windows memory -------------------------------------------------------

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
psapi.GetProcessMemoryInfo.restype = wt.BOOL
k32.GetCurrentProcess.restype = wt.HANDLE


def rss_mb():
    c = PMC()
    c.cb = ctypes.sizeof(c)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1e6, c.PeakWorkingSetSize / 1e6


# --- Python / ONNX Runtime baseline ---------------------------------------


def bench_python():
    import numpy as np
    import onnxruntime as ort

    base, _ = rss_mb()

    # Single-threaded, to match our runtime (which is single-threaded).
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    t0 = time.perf_counter()
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    t_load = time.perf_counter() - t0

    loaded, peak_loaded = rss_mb()

    feeds = {
        "input_ids": np.array([IDS], dtype=np.int64),
        "speed": np.array([1.0], dtype=np.float32),
    }

    # Warm up, then measure.
    out = sess.run(None, feeds)
    samples = out[0].ravel().shape[0]
    audio_s = samples / SAMPLE_RATE

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append(time.perf_counter() - t0)

    _, peak_after = rss_mb()
    med = statistics.median(times)

    return {
        "runtime": "python+onnxruntime",
        "load_s": t_load,
        "synth_median_s": med,
        "synth_min_s": min(times),
        "audio_s": audio_s,
        "rtf": audio_s / med,
        "rss_base_mb": base,
        "rss_loaded_mb": loaded,
        "rss_peak_mb": peak_after,
        "peak_after_load_mb": peak_loaded,
        "samples": samples,
    }


# --- Rust runtime ---------------------------------------------------------


def bench_rust():
    if not os.path.exists(RUST_EXE):
        raise SystemExit(f"missing {RUST_EXE} — run: cd runtime && cargo build --release")

    args = [RUST_EXE, MODEL] + [str(i) for i in IDS]

    # The binary prints timings and RSS to stderr; run it and parse.
    times = []
    load_s = None
    run_s = None
    samples = None
    rss_loaded = None
    rss_peak = None
    for _ in range(RUNS):
        t0 = time.perf_counter()
        p = subprocess.run(args, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        err = p.stderr
        for line in err.splitlines():
            if line.startswith("parse+topo:"):
                load_s = _parse_dur(line.split()[1])
            elif line.startswith("run:"):
                run_s = _parse_dur(line.split()[1])
            elif line.startswith("rss_after_load_mb:"):
                rss_loaded = float(line.split(":")[1])
            elif line.startswith("rss_peak_mb:"):
                rss_peak = float(line.split(":")[1])
            elif line.startswith("out dims="):
                samples = int(line.split("len=")[1].split()[0])
        times.append(run_s if run_s is not None else wall)

    audio_s = samples / SAMPLE_RATE
    med = statistics.median(times)

    return {
        "runtime": "rust (ours)",
        "load_s": load_s,
        "synth_median_s": med,
        "synth_min_s": min(times),
        "audio_s": audio_s,
        "rtf": audio_s / med,
        "rss_loaded_mb": rss_loaded,
        "rss_peak_mb": rss_peak,
        "samples": samples,
    }


def _parse_dur(s):
    """'9.2ms' / '1.23s' / '450µs' -> seconds."""
    s = s.strip()
    if s.endswith("ms"):
        return float(s[:-2]) / 1e3
    if s.endswith("µs"):
        return float(s[:-2]) / 1e6
    if s.endswith("ns"):
        return float(s[:-2]) / 1e9
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


# --- report ---------------------------------------------------------------


def report(rows):
    print()
    print("=" * 78)
    print(f"tinytts benchmark — {len(IDS)} tokens, {RUNS} runs, 1 thread")
    print("=" * 78)
    hdr = f"{'runtime':<22} {'load':>10} {'synth':>10} {'audio':>8} {'RTF':>8} {'RSS':>10}"
    print(hdr)
    print("-" * 78)
    for r in rows:
        load = f"{r['load_s']*1000:.1f} ms" if r.get("load_s") else "-"
        synth = f"{r['synth_median_s']*1000:.1f} ms"
        audio = f"{r['audio_s']:.2f} s"
        rtf = f"{r['rtf']:.1f}x"
        rss = f"{r['rss_loaded_mb']:.0f} MB" if r.get("rss_loaded_mb") else "-"
        print(f"{r['runtime']:<22} {load:>10} {synth:>10} {audio:>8} {rtf:>8} {rss:>10}")
    print("-" * 78)

    if len(rows) == 2:
        a, b = rows
        if a.get("load_s") and b.get("load_s"):
            print(f"load speedup      : {a['load_s']/b['load_s']:.1f}x")
        print(f"synth speedup     : {a['synth_median_s']/b['synth_median_s']:.2f}x")
        if a.get("rss_loaded_mb"):
            print(f"RSS reduction     : {a['rss_loaded_mb']/max(b.get('rss_loaded_mb',1),1):.1f}x")
    print()

    out = os.path.join(ROOT, "docs", "benchmark.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {out}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    rows = []
    if which in ("both", "python"):
        rows.append(bench_python())
    if which in ("both", "rust"):
        rows.append(bench_rust())
    report(rows)


if __name__ == "__main__":
    main()
