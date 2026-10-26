"""Generated sweep: proto3 vs Rust over systematic number inputs.

Covers years 1000-2100, ordinals 1-100, decimals (incl. thousands+decimal),
currency, and time formats. Any diff is a port bug. The frozen held-out set
is NOT touched.

Usage (from the repo root):
  python scripts/sweep_g2p.py
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(REPO, "g2p", "reference")
EXE = os.environ.get("G2P_EXE", r"C:\tts-target\release\g2p.exe")

sys.path.insert(0, REFERENCE)
os.chdir(REFERENCE)
import proto3  # noqa: E402

inputs = []

# years 1000-2100
for y in range(1000, 2101):
    inputs.append(f"The year was {y}.")

# ordinals 1-100
for n in range(1, 101):
    suf = "th"
    if n % 100 not in (11, 12, 13):
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    inputs.append(f"He finished {n}{suf}.")

# decimals, thousands, thousands+decimal
for w in ["0.5", "3.14", "42.50", "100.00", "1,284", "12,000", "1,284.67",
          "45,000", "999,999", "1,000,000", "7.2", "3.8", "4.0", "0.001"]:
    inputs.append(f"The value is {w} exactly.")

# currency
for w in ["$18", "$4.7", "$1,284.67", "$0.99", "$100", "$45,000"]:
    inputs.append(f"It costs {w} today.")

# time
for w in ["9:30", "6:45", "12:00", "3:05", "11:59", "7:22", "2:25", "6:07"]:
    inputs.append(f"The train leaves at {w} sharp.")

# dates
for w in ["12/3/1905", "6/19/1924", "11/8/1998", "1/1/2000", "7/4/1776"]:
    inputs.append(f"It happened on {w} that day.")

# negatives and mixed
for w in ["-7", "-42", "-1,000"]:
    inputs.append(f"The temperature was {w} degrees.")

diffs = 0
for t in inputs:
    pp, po = proto3.g2p(t)
    try:
        p = subprocess.run([EXE, t], capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        print(f"FATAL: rust HANG on: {t!r}")
        sys.exit(2)
    rp = p.stdout.decode("utf-8").rstrip("\n")
    err = p.stderr.decode("utf-8").strip()
    ro = []
    if err.startswith("warning: OOV words: "):
        ro = err[len("warning: OOV words: "):].split(", ")
    if pp != rp or set(po) != set(ro):
        diffs += 1
        if diffs <= 30:
            print(f"DIFF: {t!r}")
            print(f"  py: {pp!r}")
            print(f"  rs: {rp!r}")
            if set(po) != set(ro):
                print(f"  oov py={sorted(po)} rs={sorted(ro)}")

print(f"\nsweep inputs: {len(inputs)}, diffs: {diffs}")
sys.exit(1 if diffs else 0)
