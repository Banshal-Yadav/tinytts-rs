"""Per-op GFLOP/s table: measured ms vs exact MAC counts, against the roofline.

MAC counts come from the runtime's own `--shapes` dump (every node's real input
and output shapes at the chosen input length), not from a hand-written table.

Usage:  python scripts/bench/gflops.py [--input mid]
"""
import argparse
import json
import os
import sys
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _ids
MODEL = os.path.join(ROOT, "paradee_int8.onnx")
RT = os.path.join(ROOT, "runtime")
EXE = os.environ.get("CARGO_TARGET_DIR", os.path.join(ROOT, "runtime", "target")) + r"\release\tinytts.exe"

# AMD Ryzen 7 7735HS, AVX2+FMA: 8 f32 lanes x 2 flops x 2 FMA units = 32
# flops/cycle. The sustained clock under load is measured by
# `cargo run --release --example roofline`; it drifts with thermal state, so
# override it here rather than trusting a single reading.
CLOCK_GHZ = float(os.environ.get("CLOCK_GHZ", "4.4"))


def run(extra, ids):
    p = subprocess.run([EXE, MODEL] + extra + [str(i) for i in ids],
                       capture_output=True, text=True, cwd=RT)
    return p.stdout, p.stderr


def dims(s):
    if s in ("-", "?", ""):
        return None
    try:
        return [int(x) for x in s.split("x")]
    except ValueError:
        return None


def macs_from_shapes(stdout):
    """op -> total MACs, from the --shapes dump."""
    macs = {}
    for line in stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        op, name, ins, outs = parts
        iv = [dims(x) for x in ins.split(",")]
        ov = [dims(x) for x in outs.split(",")]
        m = 0
        if op == "Conv" and len(iv) >= 2 and iv[0] and iv[1] and ov and ov[0]:
            w = iv[1]
            if len(w) == 3:
                cout, cin, kw = w
                # Use the OUTPUT length: for a strided conv the input length is
                # not the number of output positions.
                lout = ov[0][-1]
                m = lout * cout * cin * kw
        elif op == "ConvTranspose" and len(iv) >= 2 and iv[0] and iv[1]:
            x, w = iv[0], iv[1]
            if len(x) == 3 and len(w) == 3:
                cin, cout, kw = w
                m = x[2] * cin * cout * kw
        elif op == "MatMul" and len(iv) >= 2 and iv[0] and iv[1]:
            a, b = iv[0], iv[1]
            if len(a) >= 2 and len(b) >= 2:
                m = a[-2] * a[-1] * b[-1]
        elif op == "Gemm" and len(iv) >= 2 and iv[0] and iv[1]:
            a, b = iv[0], iv[1]
            if a and b and len(a) >= 2 and len(b) >= 2:
                m = a[0] * a[1] * b[1]
        elif op == "LSTM" and len(iv) >= 3 and iv[0] and iv[1] and iv[2]:
            x, w, r = iv[0], iv[1], iv[2]
            if len(x) == 3 and len(w) == 3 and len(r) == 3:
                seq, dirs, g4, inp, h = x[0], w[0], w[1], w[2], r[2]
                m = dirs * seq * g4 * (inp + h)
        if m:
            macs[op] = macs.get(op, 0) + m
    return macs


def parse_per_op(text):
    out = {}
    in_block = False
    for line in text.splitlines():
        if "per-op-type" in line:
            in_block = True
            continue
        if in_block:
            if not line.strip():
                break
            m = re.match(r"\s*(\S+)\s+([\d.]+)\s+([\d.]+)%\s+(\d+)\s*$", line)
            if m:
                out[m.group(1)] = (float(m.group(2)), int(m.group(4)))
    return out


def parse_top_nodes(text):
    out = []
    in_block = False
    for line in text.splitlines():
        if "top 15 individual nodes" in line:
            in_block = True
            continue
        if in_block:
            m = re.match(r"\s*(\S+)\s+([\d.]+)\s+([\d.]+)%\s+(.*)$", line)
            if m:
                out.append((float(m.group(2)), m.group(1), m.group(4).strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="mid", choices=["short", "mid", "long"])
    args = ap.parse_args()
    ids = _ids.ids(args.input)

    shapes_out, _ = run(["--shapes"], ids)
    _, prof = run(["--profile"], ids)
    per_op = parse_per_op(prof)
    top = parse_top_nodes(prof)
    macs = macs_from_shapes(shapes_out)

    print(f"input={args.input} tokens={len(ids)}   clock assumption {CLOCK_GHZ} GHz "
          f"(override with CLOCK_GHZ; percentages of peak move with thermal state)")
    print()
    print(f"{'op':16} {'ms':>8} {'n':>5} {'GFLOP':>8} {'GFLOP/s':>9}")
    total_ms = 0.0
    total_gflop = 0.0
    for op in ("Conv", "MatMul", "ConvTranspose", "LSTM", "Gemm"):
        if op not in per_op:
            continue
        ms, n = per_op[op]
        gflop = macs.get(op, 0) * 2 / 1e9
        gfs = gflop / (ms / 1e3) if ms > 0 else 0.0
        total_ms += ms
        total_gflop += gflop
        print(f"{op:16} {ms:8.1f} {n:5} {gflop:8.3f} {gfs:9.1f}")
    if total_ms:
        print(f"{'total (flop ops)':16} {total_ms:8.1f} {'':5} {total_gflop:8.3f} "
              f"{total_gflop/(total_ms/1e3):9.1f}")

    print()
    print("non-flop ops (memory-bound):")
    for op in ("Mul", "Add", "SineExcitation", "Transpose", "Slice", "Pad", "Div", "Sub"):
        if op in per_op:
            ms, n = per_op[op]
            print(f"  {op:16} {ms:8.1f} ms  n={n}")

    print()
    print("top individual nodes:")
    for ms, op, name in top[:12]:
        print(f"  {ms:7.2f} ms  {op:18} {name}")


if __name__ == "__main__":
    main()
