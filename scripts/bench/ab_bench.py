"""Interleaved A/B benchmark: two binaries, alternating A,B,A,B in one session.

Absolute ms on this laptop drift by 20%+ between sessions (thermal/background),
so a before/after comparison across sessions is meaningless. This runs both
binaries in the SAME session, alternating, and reports only the RATIO.

Usage:
    python scripts/bench/ab_bench.py --a <exeA> --b <exeB> [--rounds 20] [--input mid]
    python scripts/bench/ab_bench.py --a-commit f92967f --b-commit HEAD [--rounds 20]

--a-commit/--b-commit build each commit into its own CARGO_TARGET_DIR via
`git worktree`, so the two binaries are built from source and never overwrite
each other. Worktrees land in .abwork/ (gitignored).

Reported: p50 ratio B/A and p95 ratio B/A. < 1.0 means B is faster.
Absolute ms are printed for context only.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _ids
MODEL = os.path.join(ROOT, "paradee_int8.onnx")
IDS = _ids.all_ids()
WORK = os.path.join(ROOT, ".abwork")


def build_commit(commit, tag):
    """Build `commit` in an isolated worktree + target dir. Returns exe path.

    `commit` may be `WORKTREE`, which builds the main repo's current working tree
    (including uncommitted changes) into the tag's target dir. Anything else is a
    git revision, resolved in the MAIN repo and checked out in the worktree, so a
    stale worktree from an earlier run cannot silently supply the wrong binary.
    (An earlier version reused the worktree as-is, which made `--b-commit HEAD`
    resolve to the worktree's old checkout and report a bogus ratio.)
    """
    tgt = os.path.join(WORK, f"target-{tag}")
    os.makedirs(WORK, exist_ok=True)
    env = dict(os.environ, CARGO_TARGET_DIR=tgt)
    if commit == "WORKTREE":
        # The tag's target dir may hold a build from a different source path (a
        # worktree), and cargo's fingerprint would then skip the rebuild. Clear
        # it so the working tree is always compiled fresh.
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


def run_once(exe, ids):
    p = subprocess.run([exe, MODEL, "--profile"] + [str(i) for i in ids],
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    for line in out.splitlines():
        if "synth" in line and ":" in line:
            try:
                return float(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    raise RuntimeError(f"no synth line from {exe}\n{out[-2000:]}")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="baseline exe")
    ap.add_argument("--b", help="candidate exe")
    ap.add_argument("--a-commit", help="baseline commit (built in a worktree)")
    ap.add_argument("--b-commit", help="candidate commit (built in a worktree)")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--input", default="mid", choices=["short", "mid", "long"])
    args = ap.parse_args()

    exe_a = args.a or build_commit(args.a_commit, "a")
    exe_b = args.b or build_commit(args.b_commit, "b")
    ids = IDS[args.input]

    print(f"A = {exe_a}")
    print(f"B = {exe_b}")
    print(f"input={args.input} tokens={len(ids)} rounds={args.rounds} warmup={args.warm}")
    print("interleaved A,B,A,B ... one session, background apps must be closed\n")

    for _ in range(args.warm):
        run_once(exe_a, ids)
        run_once(exe_b, ids)

    ta, tb = [], []
    t0 = time.time()
    for r in range(args.rounds):
        ta.append(run_once(exe_a, ids))
        tb.append(run_once(exe_b, ids))
        if (r + 1) % 5 == 0:
            print(f"  round {r+1}/{args.rounds}  "
                  f"A p50={statistics.median(ta):8.1f}  B p50={statistics.median(tb):8.1f}  "
                  f"ratio={statistics.median(tb)/statistics.median(ta):.4f}")
    wall = time.time() - t0

    a50, b50 = statistics.median(ta), statistics.median(tb)
    a95, b95 = pct(ta, 0.95), pct(tb, 0.95)
    print(f"\n{'':10} {'p50':>10} {'p95':>10} {'min':>10} {'max':>10}")
    print(f"{'A':10} {a50:10.1f} {a95:10.1f} {min(ta):10.1f} {max(ta):10.1f}")
    print(f"{'B':10} {b50:10.1f} {b95:10.1f} {min(tb):10.1f} {max(tb):10.1f}")
    print(f"\nRATIO B/A  p50 = {b50/a50:.4f}   p95 = {b95/a95:.4f}   "
          f"({'B faster' if b50 < a50 else 'B slower'} by {abs(1-b50/a50)*100:.1f}%)")
    print(f"absolute ms are indicative only; wall {wall:.0f}s")

    out = {
        "input": args.input, "tokens": len(ids), "rounds": args.rounds,
        "a": {"exe": exe_a, "p50": a50, "p95": a95, "min": min(ta), "max": max(ta)},
        "b": {"exe": exe_b, "p50": b50, "p95": b95, "min": min(tb), "max": max(tb)},
        "ratio_p50": b50 / a50, "ratio_p95": b95 / a95,
    }
    path = os.path.join(ROOT, "docs", "ab_last.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
