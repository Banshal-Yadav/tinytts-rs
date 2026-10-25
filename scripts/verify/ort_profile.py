"""ORT op-by-op profile on the 54-token input, single thread.

Runs ORT with enable_profiling=True, parses the JSON trace, and aggregates
per-op-type time. This is the reference side of the deficit table: it tells us
which ops ORT is fast at, without guessing.

Usage:  python scripts/verify/ort_profile.py [short|mid|long]
"""
import json
import os
import sys
import collections

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _ids
MODEL = os.path.join(ROOT, "paradee_int8.onnx")
IDS = _ids.all_ids()

which = sys.argv[1] if len(sys.argv) > 1 else "mid"
ids = IDS[which]

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.enable_profiling = True
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
feed = {
    "input_ids": np.array([ids], dtype=np.int64),
    "speed": np.array([1.0], dtype=np.float32),
}

# warmup
for _ in range(3):
    sess.run(None, feed)

prof = sess.end_profiling()
events = json.load(open(prof))
os.remove(prof)

# Aggregate kernel time per op type. ORT emits one "<node>_kernel_time" event
# per node execution; the op type is in args["op_name"] when present, otherwise
# it has to be recovered from the node name.
by_op = collections.defaultdict(lambda: [0.0, 0])
total = 0.0
for e in events:
    if e.get("cat") != "Node":
        continue
    nm = e.get("name", "")
    if not nm.endswith("_kernel_time"):
        continue
    a = e.get("args", {})
    op = a.get("op_name")
    if not op:
        # Fall back to the node name with the suffix stripped.
        op = nm[: -len("_kernel_time")]
    us = float(e.get("dur", 0))
    by_op[op][0] += us
    by_op[op][1] += 1
    total += us

print(f"ORT {ort.__version__}  input={which} tokens={len(ids)}  threads=1")
print(f"total kernel time: {total / 1000:.2f} ms")
print()
print(f"{'op':<24}{'ms':>10}{'%':>8}{'n':>7}")
for op, (us, n) in sorted(by_op.items(), key=lambda kv: -kv[1][0]):
    print(f"{op:<24}{us / 1000:>10.2f}{us / total * 100:>7.1f}%{n:>7}")

# Save for the comparison table.
out = {op: {"ms": us / 1000, "n": n} for op, (us, n) in by_op.items()}
out["__total__"] = {"ms": total / 1000, "n": sum(v[1] for v in by_op.values())}
json.dump(out, open(os.path.join(ROOT, "docs", f"ort_profile_{which}.json"), "w"), indent=1)
print()
print(f"wrote docs/ort_profile_{which}.json")
