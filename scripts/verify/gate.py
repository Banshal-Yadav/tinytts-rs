"""Run the differential gate on all three input lengths.

Usage:  python scripts/verify/gate.py            # all three
        python scripts/verify/gate.py short      # one

Requires golden_short/, golden/, golden_long/ (see scripts/verify/dump_golden.py).
Exit code 0 only if every length passes corr >= 0.99999.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNTIME = os.path.join(ROOT, "runtime")

# name -> (golden dir, min correlation)
CASES = {
    "short": ("golden_short", 0.99999),
    "mid": ("golden", 0.99999),
    "long": ("golden_long", 0.99999),
}


def run(name: str) -> bool:
    gdir, min_corr = CASES[name]
    env = dict(os.environ, GOLDEN=f"../{gdir}")
    p = subprocess.run(
        ["cargo", "test", "--release", "--test", "diff_nodes", "full_graph", "--", "--nocapture"],
        cwd=RUNTIME, env=env, capture_output=True, text=True,
    )
    out = p.stdout + p.stderr
    m = re.search(r"WAVEFORM \S+: n=(\d+)\s+max_abs=([\d.e+-]+)\s+corr=([\d.]+)", out)
    if not m:
        print(f"{name:6} FAIL  (no waveform line)")
        print(out[-2000:])
        return False
    n, max_abs, corr = int(m.group(1)), float(m.group(2)), float(m.group(3))
    ok = corr >= min_corr
    print(f"{name:6} {'PASS' if ok else 'FAIL'}  n={n:>7}  max_abs={max_abs:.3e}  corr={corr:.9f}")
    return ok


def main() -> int:
    names = sys.argv[1:] or list(CASES)
    bad = [n for n in names if not run(n)]
    print()
    print("GATE " + ("PASS" if not bad else f"FAIL: {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
