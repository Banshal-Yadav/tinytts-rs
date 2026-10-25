"""Fresh-process peak RSS + peak LIVE for our runtime, one process per input.

Each measurement spawns a brand-new process that does exactly one thing: load,
run once, report. Nothing else runs before it in that process, so the RSS
high-water mark is not contaminated by an earlier run.

The three inputs are the ones the golden dumper uses, read from the golden
manifests so this script and the differential test always agree.

Usage:  python scripts/bench/rss_fresh.py [runs]
"""
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXE = os.environ.get("CARGO_TARGET_DIR", os.path.join(ROOT, "runtime", "target")) + r"\release\examples\rssone.exe"

# name -> golden dir, in increasing input length.
CASES = [("short", "golden_short"), ("mid", "golden"), ("long", "golden_long")]

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def ids_for(gdir):
    path = os.path.join(ROOT, gdir, "manifest.json")
    if not os.path.exists(path):
        sys.exit(f"{path} missing. Run: python scripts/verify/dump_golden.py paradee_int8.onnx {gdir}")
    return json.load(open(path))["input_ids"]


def one(ids):
    env = dict(os.environ, MODEL=os.path.join(ROOT, "paradee_int8.onnx"))
    p = subprocess.run([EXE] + [str(i) for i in ids], capture_output=True, text=True, env=env)
    out = p.stdout + p.stderr
    d = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            try:
                d[k.strip()] = float(v.strip().split()[0])
            except ValueError:
                pass
    return d


def main():
    print(f"fresh process per run, {RUNS} runs each")
    print(f"{'input':6} {'tokens':>6} {'loaded':>8} {'peak LIVE':>10} {'peak RSS':>9} {'synth p50':>10}")
    for name, gdir in CASES:
        ids = ids_for(gdir)
        rows = [one(ids) for _ in range(RUNS)]
        loaded = statistics.median(r["loaded_rss_mb"] for r in rows)
        live = statistics.median(r["peak_live_mb"] for r in rows)
        rss = statistics.median(r["peak_rss_mb"] for r in rows)
        synth = statistics.median(r["synth_ms"] for r in rows)
        print(f"{name:6} {len(ids):6} {loaded:8.1f} {live:10.1f} {rss:9.1f} {synth:10.1f}")


if __name__ == "__main__":
    main()
