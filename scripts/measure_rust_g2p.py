"""Measure the Rust g2p CLI against the frozen held-out set.

Reports the same metrics as scripts/baseline_heldout.py (exact %, word
match %, PER, OOV), plus a digit-free-subset breakdown: sentences without
any digit are fully covered by the ported pipeline, so their numbers are
the honest ported-quality signal until number expansion lands.
"""
import difflib
import json
import os
import re
import subprocess
import sys
import time

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.environ.get("G2P_EXE", r"C:\tts-target\release\g2p.exe")

refs = json.load(open(os.path.join(REPO, "heldout_ref.json"), encoding="utf-8"))
num_pat = re.compile(r"\d")


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


def score(subset, label):
    exact = 0
    total_ref = 0
    total_err = 0
    word_match = 0
    oov_count = 0
    mismatches = {}
    strip = ".,!?;:\"\u201c\u201d\u2018\u2019\u2026\u2014"
    total_words = sum(
        len([w.strip(strip) for w in r["phonemes"].split() if w.strip(strip)])
        for r in subset
    )
    total_chars = sum(len(r["phonemes"]) for r in subset)

    for r in subset:
        try:
            p = subprocess.run([EXE, r["text"]], capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            print(f"FATAL: rust HANG on: {r['text']!r}")
            sys.exit(2)
        got = p.stdout.decode("utf-8").rstrip("\n")
        oov_line = p.stderr.decode("utf-8").strip()
        oov = 1 if oov_line.startswith("warning: OOV") else 0
        oov_count += oov
        ref = r["phonemes"]
        rw = [w.strip(strip) for w in ref.split() if w.strip(strip)]
        gw = [w.strip(strip) for w in got.split() if w.strip(strip)]

        if oov:
            total_err += len(ref)
        else:
            total_err += lev(got, ref)

        if not oov and got == ref:
            exact += 1
            word_match += len(rw)
        else:
            sm = difflib.SequenceMatcher(a=gw, b=rw, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    word_match += j2 - j1
                elif tag == "replace":
                    for k in range(min(j2 - j1, i2 - i1)):
                        key = (rw[j1 + k], gw[i1 + k])
                        mismatches[key] = mismatches.get(key, 0) + 1

    n = len(subset)
    print(f"--- {label}: {n} sentences ---")
    print(f"exact sentence match: {exact}/{n} = {100 * exact / n:.1f}%")
    print(f"word-level phoneme match: {word_match}/{total_words} = "
          f"{100 * word_match / max(1, total_words):.2f}%")
    print(f"PER (all, OOV = full error): {100 * total_err / max(1, total_chars):.2f}%")
    print(f"sentences with OOV: {oov_count}")
    return mismatches


t0 = time.perf_counter()
all_mismatches = score(refs, "ALL sentences")
dt = time.perf_counter() - t0

digit_free = [r for r in refs if not num_pat.search(r["text"])]
df_mismatches = score(digit_free, "digit-free subset (fully covered by the port)")

print(f"wall time: {dt * 1000 / len(refs):.2f} ms/sentence (subprocess overhead included)")
print("top word mismatches, digit-free subset (ref <- got):")
for (rr, g), c in sorted(df_mismatches.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {c:3d}  {rr!r} <- {g!r}")
