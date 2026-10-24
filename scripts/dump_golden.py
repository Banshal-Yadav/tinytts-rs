"""Dump every intermediate tensor of the ONNX graph as golden files.

Runs ONNX Runtime once on a fixed input with ALL node outputs exposed as graph
outputs, then writes:
  golden/t<NNNN>.npy       one file per tensor, named by manifest index
  golden/manifest.json     node order, op types, names, input/output tensor names,
                           per-tensor dtype + shape, and the fixed input

Tensor files are indexed (t0000.npy), NOT named after the tensor, so that
sanitisation can never collide (`/F0.1/Mul` vs `/F0/1/Mul`). The real name lives
in the manifest.

Initializers are NOT dumped: they are weights/constants that the Rust runtime
loads straight from the model.

The Rust differential test (runtime/tests/diff_nodes.rs) consumes only these
files, so it runs without Python or onnxruntime.

Usage:  python scripts/dump_golden.py [model.onnx] [out_dir]
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort

MODEL = sys.argv[1] if len(sys.argv) > 1 else "paradee_int8.onnx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "golden"

# Fixed input. Override with IDS="1,2,3" in the environment.
# Default is the 52-token "Hello, this is a test of the speech synthesis system."
# phoneme sequence (IPA, mapped through the model's vocab).
DEFAULT_IDS = [
    0, 50, 86, 54, 57, 135, 3, 16, 81, 102, 61, 16, 102, 68, 16, 83, 16, 62,
    86, 61, 62, 16, 71, 64, 16, 81, 83, 16, 61, 58, 51, 158, 62, 131, 16, 61,
    102, 56, 119, 83, 61, 102, 61, 16, 61, 102, 61, 62, 83, 55, 4, 0,
]
IDS = [int(x) for x in os.environ["IDS"].split(",")] if "IDS" in os.environ else DEFAULT_IDS
SPEED = float(os.environ.get("SPEED", "1.0"))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    m = onnx.load(MODEL)
    g = m.graph

    init_names = {i.name for i in g.initializer}

    # Every node output becomes a graph output so ORT materialises it.
    # Skip initializers (ORT rejects a graph output that is also an initializer).
    existing = {o.name for o in g.output}
    for n in g.node:
        for o in n.output:
            if o and o not in existing and o not in init_names:
                existing.add(o)
                vi = onnx.helper.ValueInfoProto()
                vi.name = o
                g.output.append(vi)

    probe = os.path.join(OUT, "_probe.onnx")
    onnx.save(m, probe)
    sess = ort.InferenceSession(probe, providers=["CPUExecutionProvider"])

    out_names = [o.name for o in sess.get_outputs()]
    feeds = {
        "input_ids": np.array([IDS], dtype=np.int64),
        "speed": np.array([SPEED], dtype=np.float32),
    }
    results = sess.run(out_names, feeds)

    # Indexed filenames: no sanitisation, no collisions.
    tensors = {}
    for i, (name, arr) in enumerate(zip(out_names, results)):
        fn = f"t{i:04d}.npy"
        np.save(os.path.join(OUT, fn), arr)
        tensors[name] = {
            "file": fn,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }

    nodes = [
        {
            "index": i,
            "op": n.op_type,
            "name": n.name,
            "inputs": list(n.input),
            "outputs": list(n.output),
        }
        for i, n in enumerate(g.node)
    ]

    manifest = {
        "model": os.path.basename(MODEL),
        "input_ids": IDS,
        "speed": SPEED,
        "num_nodes": len(nodes),
        "num_initializers": len(init_names),
        "op_counts": dict(Counter(n.op_type for n in g.node)),
        "nodes": nodes,
        "tensors": tensors,
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"nodes={len(nodes)} inits={len(init_names)} tensors={len(tensors)}")
    print("op_counts:")
    for op, c in sorted(manifest["op_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {op:28s} {c}")


if __name__ == "__main__":
    main()
