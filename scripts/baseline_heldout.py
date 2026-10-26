"""Reference baseline: the repo's proto3.py (g2p/reference) on the frozen
held-out set.

Reports: exact sentence match %, word-level phoneme match %, PER (all and
non-OOV), OOV count + word list.
"""
import difflib, json, os, re, sys, time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "g2p", "reference"))
import proto3

refs = json.load(open(os.path.join(REPO, "heldout_ref.json"), encoding="utf-8"))
n = len(refs)

exact = 0
total_ref = 0
total_err = 0
noov_ref = 0
noov_err = 0
oov_words = Counter()
word_total = 0
word_match = 0
mismatched_words = Counter()

def lev(a, b):
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]

t0 = time.perf_counter()
for r in refs:
    text, ref = r["text"], r["phonemes"]
    got, oov = proto3.g2p(text)
    if oov:
        for w in oov:
            oov_words[w] += 1
    if not oov and got == ref:
        exact += 1
        continue
    # PER (OOV sentences count as full error, same as the spike)
    total_ref += len(ref)
    total_err += len(ref) if oov else lev(got, ref)
    if not oov:
        noov_ref += len(ref)
        noov_err += lev(got, ref)
        # word-level: split both by spaces, strip trailing punctuation
        # align word lists (token counts can differ on number expansions)
        strip = '.,!?;:"\u201c\u201d\u2018\u2019\u2026\u2014'
        gw = [w.strip(strip) for w in got.split()]
        rw = [w.strip(strip) for w in ref.split()]
        sm = difflib.SequenceMatcher(a=gw, b=rw, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                word_total += j2 - j1
                word_match += j2 - j1
            elif tag == "replace":
                word_total += max(j2 - j1, i2 - i1)
                for k in range(min(j2 - j1, i2 - i1)):
                    mismatched_words[(rw[j1 + k], gw[i1 + k])] += 1
            elif tag == "delete":
                word_total += i2 - i1
            elif tag == "insert":
                word_total += j2 - j1
dt = time.perf_counter() - t0

print(f"sentences: {n}")
print(f"exact sentence match: {exact}/{n} = {100 * exact / n:.1f}%")
print(f"word-level phoneme match: {word_match}/{word_total} = {100 * word_match / max(1, word_total):.2f}%")
print(f"PER (all, OOV = full error): {100 * total_err / max(1, total_ref):.2f}%")
print(f"PER (non-OOV sentences only): {100 * noov_err / max(1, noov_ref):.2f}%")
print(f"OOV: {sum(oov_words.values())} tokens, {len(oov_words)} unique")
print(f"OOV words: {sorted(oov_words)}")
print(f"g2p speed: {dt * 1000 / n:.2f} ms/sentence (pure python)")
print("top word mismatches (ref <- got):")
for (rr, g), c in mismatched_words.most_common(20):
    print(f"  {c:3d}  {rr!r} <- {g!r}")
