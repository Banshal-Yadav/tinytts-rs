"""Chunking defaults: peak RSS, synth time, first-audio latency, audio length.

Usage:  python scripts/chunkdefaults.py
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
    p = subprocess.run([EXE, MODEL, "--profile"] + extra + [str(i) for i in IDS],
                       capture_output=True, text=True, cwd=RT)
    L = p.stderr.splitlines()

    def g(pat):
        for l in L:
            if pat in l:
                return l.split(":", 1)[1].strip()
        return "?"

    nch = next((l for l in L if l.startswith("chunked:")), "unchunked")
    with wave.open(os.path.join(RT, "out.wav")) as w:
        aud = w.getnframes() / w.getframerate()
    return {
        "chunks": nch,
        "peak": g("rss peak"),
        "synth": g("synth (all") if "synth (all" in p.stderr else g("  synth"),
        "first": g("first audio") if "first audio" in p.stderr else g("  synth"),
        "audio": aud,
    }


def main():
    cases = [
        ("default (auto)", []),
        ("--no-chunk", ["--no-chunk"]),
        ("--chunk 100", ["--chunk", "100"]),
        ("--chunk 60", ["--chunk", "60"]),
    ]
    print(f"{'case':16} {'chunks':34} {'peak RSS':>9} {'synth':>10} {'first':>9} {'audio':>8}")
    for label, extra in cases:
        r = run(extra)
        print(f"{label:16} {r['chunks'][:34]:34} {r['peak']:>9} {r['synth']:>10} "
              f"{r['first']:>9} {r['audio']:>7.2f}s")


if __name__ == "__main__":
    main()
