"""Differential test driver: Python proto3 reference vs the Rust g2p crate.

Runs BOTH implementations over the same inputs (the 520-sentence spike
corpus plus edge-case strings) and byte-compares the outputs. Any diff is
either a port bug (must be fixed) or a number-expansion gap (expected until
the number layer is ported; reported per category).

The frozen held-out set is NOT touched here.

Usage (from the repo root):
  python scripts/diff_g2p.py
"""
import json
import os
import re
import subprocess
import sys
from collections import Counter

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
# words, stems, hyphens, punctuation corners, apostrophe corners (the
# leading-quote input is a hang regression test)
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
    "'Wait,' he whispered, 'did you hear that?'",
    "dogs' bark '",
]

# --- run both ---------------------------------------------------------------
py_out = []
for t in inputs:
    ps, oov = proto3.g2p(t)
    py_out.append((ps, oov))

rs_out = []
for i, t in enumerate(inputs):
    try:
        p = subprocess.run([EXE, t], capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        print(f"FATAL: rust HANG (timeout 30s) on input {i}: {t!r}")
        sys.exit(2)
    ps = p.stdout.decode("utf-8").rstrip("\n")
    err = p.stderr.decode("utf-8").strip()
    oov = []
    if err.startswith("warning: OOV words: "):
        oov = err[len("warning: OOV words: "):].split(", ")
    rs_out.append((ps, oov))

# --- compare ----------------------------------------------------------------
num_pat = re.compile(r"\d")
real_diffs = 0
num_diffs = 0
categories = Counter()


def number_category(text):
    """Classify a number-bearing input by its number pattern."""
    cats = set()
    for tok in re.findall(r"\d+(?:[.,]\d+)*(?::\d+)?(?:st|nd|rd|th)?", text):
        if re.fullmatch(r"\d{4}", tok) and 1000 <= int(tok) <= 2099:
            cats.add("year")
        elif tok.endswith(("st", "nd", "rd", "th")):
            cats.add("ordinal")
        elif re.fullmatch(r"\d+:\d+", tok):
            cats.add("time")
        elif re.fullmatch(r"\d+(\.\d+)+", tok):
            cats.add("decimal")
        elif re.fullmatch(r"\d+(,\d{3})+", tok):
            cats.add("thousands")
        else:
            cats.add("plain")
    if "%" in text:
        cats.add("percent")
    if "$" in text:
        cats.add("currency")
    return cats or {"plain"}


for t, (pp, po), (rp, ro) in zip(inputs, py_out, rs_out):
    if pp == rp and set(po) == set(ro):
        continue
    if num_pat.search(t):
        num_diffs += 1
        for c in number_category(t):
            categories[c] += 1
    else:
        real_diffs += 1
        print(f"REAL DIFF (no digits): {t!r}")
        print(f"  py: {pp!r}")
        print(f"  rs: {rp!r}")
        print(f"  oov py={sorted(po)} rs={sorted(ro)}")

n = len(inputs)
print(f"\ninputs: {n}")
print(f"byte-exact matches: {n - real_diffs - num_diffs}")
print(f"number-scope diffs (expected until number expansion is ported): {num_diffs}")
print(f"REAL diffs (port bugs): {real_diffs}")
print("number diff categories (inputs may hit several):")
for c, k in categories.most_common():
    print(f"  {c:10s} {k}")
sys.exit(1 if real_diffs else 0)
