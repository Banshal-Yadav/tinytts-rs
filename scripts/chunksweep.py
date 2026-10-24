"""Chunk-size sweep: peak RSS, synth time, first-audio latency, and audio length.

The audio-length column matters: each chunk is a separate utterance, so the
model's duration predictor gives it its own pacing and the total audio grows.
This script quantifies that cost alongside the memory win.

Usage:  python scripts/chunksweep.py
"""
import json
import os
import sys
import subprocess
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import _ids
EXE = os.environ.get("CARGO_TARGET_DIR", os.path.join(ROOT, "runtime", "target")) + r"\release\tinytts.exe"
MODEL = os.path.join(ROOT, "paradee_int8.onnx")
RT = os.path.join(ROOT, "runtime")
IDS = _ids.ids("long")


def run(extra):
    p = subprocess.run(
        [EXE, MODEL] + extra + [str(i) for i in IDS],
        capture_output=True, text=True, cwd=RT,
    )
    lines = p.stderr.splitlines()
    out = {}
    for l in lines:
        if l.startswith("chunked:"):
            out["chunks"] = int(l.split()[1])
        if l.startswith("rss_peak_mb"):
            out["peak"] = float(l.split(":")[1])
        if "synth (all" in l:
            out["synth"] = float(l.split(":")[1].split()[0])
        if "first audio" in l:
            out["first"] = float(l.split(":")[1].split()[0])
        if l.strip().startswith("audio ") and "s" in l:
            out["audio"] = float(l.split(":")[1].split()[0])
        if l.startswith("synth ") and "ms" in l:
            out["synth"] = float(l.split(":")[1].split()[0])
        if l.startswith("  synth ") and "ms" in l:
            out["synth"] = float(l.split(":")[1].split()[0])
        if l.startswith("  audio ") and "s" in l:
            out["audio"] = float(l.split(":")[1].split()[0])
    return out


def main():
    base = run(["--profile"])
    print(f"unchunked: peak={base['peak']:.1f} MB  synth={base['synth']:.0f} ms  "
          f"audio={base['audio']:.2f} s")
    print()
    print(f"{'max':>5} {'chunks':>7} {'peak RSS':>9} {'synth':>9} {'first':>8} "
          f"{'audio':>8} {'audio growth':>13}")
    for mx in (30, 40, 50, 60, 80, 100):
        r = run(["--chunk", str(mx)])
        growth = (r["audio"] / base["audio"] - 1.0) * 100.0
        print(f"{mx:>5} {r['chunks']:>7} {r['peak']:>8.1f}M {r['synth']:>8.0f}m "
              f"{r['first']:>7.0f}m {r['audio']:>7.2f}s {growth:>12.1f}%")


if __name__ == "__main__":
    main()
