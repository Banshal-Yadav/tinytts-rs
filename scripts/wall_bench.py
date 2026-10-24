"""Three-way interleaved wall-clock benchmark.

Runs N binaries alternately (A,B,C,A,B,C,...) in one session and reports wall
p50/p95 for each, plus the ratios against the first. Absolute ms drift 20%+
between sessions on this laptop, so only the ratios are meaningful.

Wall time is the runtime's own `wall (load+synth)` line, which is measured
inside the process with `Instant`. Per-node profile timing is NOT used.

Usage:
    python scripts/wall_bench.py --commits f92967f a6d76b5 HEAD --rounds 20
    python scripts/wall_bench.py --exes a.exe b.exe --rounds 20
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import _ids
MODEL = os.path.join(ROOT, "paradee_int8.onnx")
IDS = _ids.all_ids()
WORK = os.path.join(ROOT, ".abwork")


def build(commit, tag):
    tgt = os.path.join(WORK, f"target-{tag}")
    os.makedirs(WORK, exist_ok=True)
    env = dict(os.environ, CARGO_TARGET_DIR=tgt)
    if commit == "WORKTREE":
        if os.path.isdir(tgt):
            shutil.rmtree(tgt, ignore_errors=True)
        subprocess.run(["cargo", "build", "--release"], cwd=os.path.join(ROOT, "runtime"),
                       env=env, check=True, capture_output=True)
        exe = os.path.join(tgt, "release", "tinytts.exe")
        print(f"  {tag}: WORKTREE  {os.path.getsize(exe)} bytes")
        return exe
    wt = os.path.join(WORK, tag)
    if not os.path.isdir(wt):
        subprocess.run(["git", "worktree", "add", "--detach", wt, commit],
                       cwd=ROOT, check=True, capture_output=True)
    sha = subprocess.run(["git", "rev-parse", commit], cwd=ROOT,
                         check=True, capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "checkout", "--detach", "--force", sha], cwd=wt,
                   check=True, capture_output=True)
    subprocess.run(["cargo", "build", "--release"], cwd=os.path.join(wt, "runtime"),
                   env=env, check=True, capture_output=True)
    exe = os.path.join(tgt, "release", "tinytts.exe")
    print(f"  {tag}: {commit} -> {sha[:9]}  {os.path.getsize(exe)} bytes")
    return exe


def wall_ms(exe, ids):
    p = subprocess.run([exe, MODEL, "--profile"] + [str(i) for i in ids],
                       capture_output=True, text=True)
    for line in (p.stdout + p.stderr).splitlines():
        if "wall (load+synth)" in line:
            return float(line.split(":")[1].split()[0])
    raise RuntimeError(f"no wall line from {exe}\n{(p.stdout + p.stderr)[-1500:]}")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commits", nargs="+")
    ap.add_argument("--exes", nargs="+")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--input", default="mid", choices=["short", "mid", "long"])
    args = ap.parse_args()

    if args.exes:
        exes = args.exes
        labels = [os.path.basename(e) for e in exes]
    else:
        labels = args.commits
        exes = [build(c, f"w{i}") for i, c in enumerate(args.commits)]

    ids = IDS[args.input]
    print(f"input={args.input} tokens={len(ids)} rounds={args.rounds} warmup={args.warm}")
    print("interleaved round-robin, one session, background apps must be closed\n")

    for _ in range(args.warm):
        for e in exes:
            wall_ms(e, ids)

    samples = [[] for _ in exes]
    for r in range(args.rounds):
        for i, e in enumerate(exes):
            samples[i].append(wall_ms(e, ids))
        if (r + 1) % 5 == 0:
            meds = [statistics.median(s) for s in samples]
            print(f"  round {r+1}/{args.rounds}  " +
                  "  ".join(f"{labels[i]}={meds[i]:7.1f}" for i in range(len(exes))))

    print()
    print(f"{'label':>12} {'p50':>9} {'p95':>9} {'min':>9} {'max':>9} {'ratio p50':>10}")
    base = statistics.median(samples[0])
    for i, s in enumerate(samples):
        p50 = statistics.median(s)
        print(f"{labels[i]:>12} {p50:9.1f} {pct(s,0.95):9.1f} {min(s):9.1f} "
              f"{max(s):9.1f} {p50/base:10.4f}")
    print("\nratios are against the first label; absolute ms are indicative only")


if __name__ == "__main__":
    main()
