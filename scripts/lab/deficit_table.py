"""ORT-vs-ours per-op deficit table, using honest per-op timings.

Two problems with the naive comparison, both fixed here:

1. ORT's profiler inflates absolute times badly. Enabling it takes the 54-token
   run from 96 ms to 148 ms, and the kernel sum is ~12x the wall. The per-op
   *proportions* are stable to ~10% across runs, so the comparison is done on
   shares scaled to ORT's real unprofiled wall.

2. Our `--profile` wraps every node in Instant::now(), which adds ~25 ms spread
   over 2529 nodes in proportion to node count, not work. Ops with many tiny
   nodes (Cast 300, Add 105, Transpose 85, Slice 85) are inflated far more than
   ops with few big nodes (Conv 49). This script uses the thresholded per-op
   timings from examples/wallacct.rs instead, which time only ops with <= 100
   nodes.

Usage:  python scripts/lab/deficit_table.py [short|mid|long]
"""
import collections
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _ids
MODEL = os.path.join(ROOT, "paradee_int8.onnx")
IDS = _ids.all_ids()

which = sys.argv[1] if len(sys.argv) > 1 else "mid"
ids = IDS[which]

# ORT's real wall, measured without the profiler.
ORT_WALL = {"short": 34.5, "mid": 96.7, "long": 868.9}[which]

# ---- ORT ------------------------------------------------------------------
ort_json = os.path.join(ROOT, "docs", f"ort_profile_{which}.json")
if not os.path.exists(ort_json):
    print(f"missing {ort_json}; run scripts/verify/ort_profile.py {which} first")
    sys.exit(1)
ort_raw = json.load(open(ort_json))
ort_total = ort_raw.pop("__total__")["ms"]
ort = {k: v["ms"] for k, v in ort_raw.items()}

# ---- ours (thresholded per-op timings) ------------------------------------
exe = os.path.join(os.environ.get("CARGO_TARGET_DIR", r"C:\tts-target"),
                   "release", "examples", "wallacct.exe")
p = subprocess.run([exe] + [str(i) for i in ids], capture_output=True, text=True,
                   cwd=os.path.join(ROOT, "runtime"),
                   env=dict(os.environ, MODEL=MODEL))
out = p.stdout + p.stderr

ours = {}
in_ops = False
our_synth = None
our_kernel = None
our_overhead = None
for line in out.splitlines():
    if "every node timed" in line:
        in_ops = True
        continue
    if "thresholded:" in line:
        in_ops = False
        continue
    if in_ops:
        m = re.match(r"^(\S+)\s+([\d.]+)\s+[\d.]+%$", line.strip())
        if m:
            ours[m.group(1)] = float(m.group(2))
        elif line.startswith("ALL TOTAL"):
            in_ops = False
    if line.startswith("SYNTH"):
        our_synth = float(line.split()[1])
    if line.startswith("kernel (all ops)"):
        our_kernel = float(line.split()[3])
    if "of timer overhead" in line:
        our_overhead = float(line.split("->")[1].split()[0])

if not ours:
    print("could not parse our per-op table")
    print(out[-3000:])
    sys.exit(1)

# The "every node timed" pass includes ~14 ms of Instant::now() overhead spread
# over 2529 nodes in proportion to node count. Subtract it per op, using the
# node count from the graph, so the numbers are comparable to ORT's.
node_counts = collections.Counter()
for n in json.load(open(os.path.join(ROOT, "golden", "manifest.json")))["nodes"]:
    node_counts[n["op"]] += 1
# Our graph fuses some ops away; use our own counts where we have them.
OUR_COUNTS = {
    "SineExcitation": 1, "ElemChain": 89, "StridedSlice": 8, "Nop": 89,
    "Dequant": 105, "DequantAlias": 30,
}
for k, v in OUR_COUNTS.items():
    node_counts[k] = v

total_nodes = sum(node_counts[op] for op in ours)
per_node = (our_overhead or 0.0) / max(total_nodes, 1)
for op in list(ours):
    ours[op] = max(0.0, ours[op] - per_node * node_counts[op])

# Our per-op numbers now cover every node and have the timer overhead removed,
# so they sum to the kernel time directly.
timed = sum(ours.values())
our_scale = our_kernel / timed if timed > 0 else 1.0

# ---- map our fused ops onto ORT's equivalents -----------------------------
ALIAS = {
    "SineExcitation": "Sin",
    "ElemChain": "Mul",
    "StridedSlice": "Gather",
    "Dequant": "DequantizeLinear",
    "DequantAlias": "DequantizeLinear",
}

merged = collections.defaultdict(lambda: [0.0, 0.0])
for k, v in ort.items():
    merged[k][0] += v * (ORT_WALL / ort_total)
for k, v in ours.items():
    merged[ALIAS.get(k, k)][1] += v * our_scale

rows = [(k, a, b) for k, (a, b) in merged.items() if a > 0.1 or b > 0.1]
rows.sort(key=lambda r: -(r[1] - r[2]))

print(f"input={which} tokens={len(ids)}  single thread")
print(f"ORT real wall {ORT_WALL:.1f} ms (unprofiled); our synth {our_synth:.1f} ms")
print(f"our kernel {our_kernel:.1f} ms; per-op sum {timed:.1f} ms, scaled by {our_scale:.3f}")
print(f"timer overhead removed: {our_overhead:.2f} ms over {total_nodes} nodes")
print()
hdr = f"{'op':<22}{'ORT ms':>9}{'ours ms':>9}{'deficit':>9}{'ours/ORT':>10}"
print(hdr)
print("-" * len(hdr))
for k, a, b in rows:
    d = a - b
    r = (b / a) if a > 0.05 else float("nan")
    print(f"{k:<22}{a:>9.2f}{b:>9.2f}{d:>+9.2f}{r:>10.2f}")
print("-" * len(hdr))
ta = sum(a for _, a, _ in rows)
tb = sum(b for _, _, b in rows)
print(f"{'TOTAL':<22}{ta:>9.2f}{tb:>9.2f}{ta - tb:>+9.2f}{tb / ta:>10.2f}")
print()
print("deficit = ORT ms - ours ms (positive means we are slower on that op)")
print("ours/ORT < 1 means we are faster")
print()
print("NOTE: rows where ORT is ~0 are ops ORT fuses away (ReduceSum, Less, Split,")
print("BiasGelu, FastGelu) or that our graph fuses (SineExcitation, ElemChain).")
print("Those rows are not comparable and are excluded from the ranking below.")
print()
print("=== ranked by absolute deficit (comparable rows only) ===")
comp = [(k, a, b) for k, a, b in rows if a > 0.5 and b > 0.5]
comp.sort(key=lambda r: -(r[1] - r[2]))
for k, a, b in comp[:15]:
    print(f"  {k:<20} ORT {a:>7.2f}  ours {b:>7.2f}  deficit {a - b:>+7.2f}  ratio {b / a:>5.2f}")
