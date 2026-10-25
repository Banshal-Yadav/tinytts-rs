"""Measure the Rust g2p CLI against the frozen held-out set.

Builds the release CLI, runs every heldout sentence through it, and reports
the same metrics as scripts/baseline_heldout.py (exact %, word match %, PER,
OOV). This is the Day-2 lexicon-only number; it is expected to be low until
the CMU mapping and context rules land on Day 3.
"""
import json
import os
import subprocess
import sys
import time

REPO = r"C:\Users\bansa\OneDrive\Desktop\TTS-8M-model"
EXE = r"C:\tts-target\release\g2p.exe"

refs = json.load(open(os.path.join(REPO, "heldout_ref.json"), encoding="utf-8"))


def lev(a, b):
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]


exact = 0
total_ref = 0
total_err = 0
word_total = 0
word_match = 0
oov_count = 0
mismatches = {}

t0 = time.perf_counter()
for r in refs:
    p = subprocess.run([EXE, r["text"]], capture_output=True, text=True,
                       encoding="utf-8")
    got = p.stdout.strip()
    oov_line = p.stderr.strip()
    oov = 1 if "warning: OOV" in oov_line else 0
    oov_count += oov
    ref = r["phonemes"]
    if not oov and got == ref:
        exact += 1
        continue
    total_ref += len(ref)
    total_err += len(ref) if oov else lev(got, ref)
    if not oov:
        strip = ".,!?;:\"\u201c\u201d\u2018\u2019\u2026\u2014"
        gw = [w.strip(strip) for w in got.split()]
        rw = [w.strip(strip) for w in ref.split()]
        import difflib
        sm = difflib.SequenceMatcher(a=gw, b=rw, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                word_total += j2 - j1
                word_match += j2 - j1
            elif tag == "replace":
                word_total += max(j2 - j1, i2 - i1)
                for k in range(min(j2 - j1, i2 - i1)):
                    key = (rw[j1 + k], gw[i1 + k])
                    mismatches[key] = mismatches.get(key, 0) + 1
            elif tag == "delete":
                word_total += i2 - i1
            elif tag == "insert":
                word_total += j2 - j1
dt = time.perf_counter() - t0

n = len(refs)
print(f"sentences: {n}")
print(f"exact sentence match: {exact}/{n} = {100 * exact / n:.1f}%")
print(f"word-level phoneme match: {word_match}/{word_total} = "
      f"{100 * word_match / max(1, word_total):.2f}%")
print(f"PER (all, OOV = full error): {100 * total_err / max(1, total_ref):.2f}%")
print(f"sentences with OOV: {oov_count}")
print(f"wall time: {dt * 1000 / n:.2f} ms/sentence (subprocess overhead included)")
print("top word mismatches (ref <- got):")
for (rr, g), c in sorted(mismatches.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {c:3d}  {rr!r} <- {g!r}")
