"""Check the chunked output for boundary clicks and compare to the unchunked run.

Runs the runtime twice on the same ids (unchunked and --chunk N), then:
  * reports the max absolute sample-to-sample jump inside each chunk boundary
    region vs the global distribution (a click shows up as an outlier),
  * reports the correlation between the two waveforms where they overlap.

Usage:  python scripts/chunkcheck.py [max_tokens]
"""
import json
import os
import subprocess
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import _ids
EXE = os.environ.get("CARGO_TARGET_DIR", os.path.join(ROOT, "runtime", "target")) + r"\release\tinytts.exe"
MODEL = os.path.join(ROOT, "paradee_int8.onnx")
RT = os.path.join(ROOT, "runtime")
IDS = _ids.ids("long")

MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def run(extra, tag):
    p = subprocess.run(
        [EXE, MODEL] + extra + [str(i) for i in IDS],
        capture_output=True, text=True, cwd=RT,
    )
    src = os.path.join(RT, "out.wav")
    dst = os.path.join(ROOT, "audio_out", f"_chunkcheck_{tag}.wav")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.replace(src, dst)
    with wave.open(dst) as w:
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return data, p.stderr


def main():
    a, ea = run(["--no-chunk"], "plain")
    b, eb = run(["--chunk", str(MAX)], "chunked")
    print(f"unchunked: {len(a)} samples ({len(a)/24000:.2f}s)")
    print(f"chunked  : {len(b)} samples ({len(b)/24000:.2f}s)  max={MAX}")

    # Boundary clicks: the chunked run's sample-to-sample jumps, and where the
    # largest ones sit relative to the chunk boundaries.
    d = np.abs(np.diff(b))
    thr = np.percentile(d, 99.99)
    big = np.where(d > max(thr, 0.05))[0]
    print(f"\nsample-to-sample |diff|: p50={np.percentile(d,50):.5f} "
          f"p99.9={np.percentile(d,99.9):.5f} max={d.max():.5f}")
    print(f"jumps > {max(thr,0.05):.4f}: {len(big)}")
    if len(big):
        print(f"  at samples {big[:12].tolist()}")

    # Chunk boundaries from the runtime log.
    bounds = []
    for line in eb.splitlines():
        if line.strip().startswith("chunk ") and "->" in line:
            parts = line.split("->")[1].split("samples")[0].strip()
            bounds.append(int(parts))
    cum = np.cumsum(bounds)
    print(f"\nchunk boundaries (samples): {cum.tolist()}")
    for c in cum[:-1]:
        lo, hi = max(0, c - 200), min(len(d), c + 200)
        print(f"  boundary {c}: local max |diff| = {d[lo:hi].max():.5f} "
              f"(global p99.9 = {np.percentile(d,99.9):.5f})")

    # Correlation where the two overlap.
    n = min(len(a), len(b))
    aa, bb = a[:n], b[:n]
    corr = np.corrcoef(aa, bb)[0, 1]
    print(f"\ncorr(unchunked, chunked) over first {n} samples = {corr:.6f}")


if __name__ == "__main__":
    main()
