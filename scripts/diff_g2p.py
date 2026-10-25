"""Differential test driver: Python proto3 reference vs the Rust g2p crate.

Runs BOTH implementations over the same inputs (the 520-sentence spike
corpus plus a set of tricky edge-case strings) and byte-compares the
outputs. Any diff is a port bug; the frozen held-out set is NOT touched.

Usage (from the repo root):
  python scripts/diff_g2p.py            # compare + report
  python scripts/diff_g2p.py --dump     # also write diffs to a file
"""
import json
import os
import subprocess
import sys

REPO = r"C:\Users\bansa\OneDrive\Desktop\TTS-8M-model"
SPIKE = r"C:\Users\bansa\OneDrive\Desktop\g2p-spike"
EXE = r"C:\tts-target\release\g2p.exe"

sys.path.insert(0, SPIKE)
os.chdir(SPIKE)
import proto3  # noqa: E402

# --- inputs -----------------------------------------------------------------
sentences = json.load(open(os.path.join(SPIKE, "sentences.json"), encoding="utf-8"))
inputs = [s["text"] for s in sentences]

# edge cases: function words in every position, capitals, acronyms, CMU-only
# words, stems, hyphens, punctuation corners
inputs += [
    "The cat sat on the mat.",
    "the apple the dog the end",
    "A bird. A bird flew. A a A I",
    "I think that that 'That' is that.",
    "to work, to eat, to 7",
    "in there am I",
    "NASA and FBI met NATO.",
    "The aardvark ran past aardvarks.",
    "running walking making stopping cities boxes",
    "Hello There, Wait!",
    "old Old OLD old's",
    "wait, wait; wait: wait! wait? wait...",
    "e-mail me at home",
    "It's the dog's dinner.",
    "zzz qqq xox",
]

# --- run both ---------------------------------------------------------------
py_out = []
for t in inputs:
    ps, oov = proto3.g2p(t)
    py_out.append((ps, oov))

# Rust: one process, one line per input (stdin mode). Empty inputs are
# skipped by the CLI, so replace them with a sentinel.
lines = [t if t.strip() else "<EMPTY>" for t in inputs]
p = subprocess.run([EXE], input="\n".join(lines), capture_output=True,
                   text=True, encoding="utf-8", timeout=300)
rs_lines = p.stdout.split("\n")
if rs_lines and rs_lines[-1] == "":
    rs_lines.pop()
if len(rs_lines) != len(inputs):
    print(f"FATAL: rust returned {len(rs_lines)} lines for {len(inputs)} inputs")
    sys.exit(2)
# stderr OOV warnings come in input order (one per OOV-bearing line)
rs_oovs = []
for line in p.stderr.splitlines():
    if line.startswith("warning: OOV words: "):
        rs_oovs.append(set(line[len("warning: OOV words: "):].split(", ")))
rs_out = []
oov_iter = iter(rs_oovs)
for t, rl in zip(inputs, rs_lines):
    ps = rl
    if t.strip() == "<EMPTY>" and ps == "<EMPTY>":
        ps = ""
    rs_out.append((ps, next(oov_iter, set())))

# --- compare ----------------------------------------------------------------
diffs = 0
for t, (pp, po), (rp, ro) in zip(inputs, py_out, rs_out):
    # OOV comparison: Rust reports "warning: OOV words: a, b"; Python a list.
    py_oov = set(po)
    rs_oov = set()
    if ro.startswith("warning: OOV words: "):
        rs_oov = set(ro[len("warning: OOV words: "):].split(", "))
    if pp != rp or py_oov != rs_oov:
        diffs += 1
        print(f"DIFF: {t!r}")
        if pp != rp:
            print(f"  py: {pp!r}")
            print(f"  rs: {rp!r}")
            # first differing char
            for i, (a, b) in enumerate(zip(pp, rp)):
                if a != b:
                    print(f"  first diff at char {i}: py={a!r} rs={b!r}")
                    break
            else:
                print(f"  length diff: py={len(pp)} rs={len(rp)}")
        if py_oov != rs_oov:
            print(f"  oov py={sorted(py_oov)} rs={sorted(rs_oov)}")

n = len(inputs)
print(f"\ninputs: {n}, byte-exact matches: {n - diffs}, diffs: {diffs}")
sys.exit(1 if diffs else 0)
