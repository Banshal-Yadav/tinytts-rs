"""Analyze current baseline on the 525-sentence DEV set before any changes.

Reports:
  - Exact sentence match %
  - Word-level phoneme match % (fixed ref words denominator)
  - PER (all)
  - Sentences with OOV
  - Breakdown of mismatches by cause:
      * the/a/to context
      * possessive ('s missing in lexicon / dropped)
      * compound (splitting or compound plural)
      * heteronym (lead, read, wind, live, tear, bow, bass, dove, etc.)
      * POS-aware context (preposition vs particle, noun vs verb)
      * stress / reduction
      * other
"""
import difflib
import json
import os
import re
import subprocess
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.environ.get("G2P_EXE", r"C:\tts-target\release\g2p.exe")
dev_refs = json.load(open(os.path.join(REPO, "g2p", "dev", "dev_ref.json"), encoding="utf-8"))
strip = ".,!?;:\"\u201c\u201d\u2018\u2019\u2026\u2014"
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TOTAL_WORDS = sum(
    len([w.strip(strip) for w in r["phonemes"].split() if w.strip(strip)])
    for r in dev_refs
)
TOTAL_CHARS = sum(len(r["phonemes"]) for r in dev_refs)


def lev(a, b):
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]


HETERONYMS = {
    "lead", "read", "wind", "live", "tear", "bow", "bass", "dove",
    "close", "record", "present", "object", "subject", "produce",
    "project", "refuse", "content", "wound", "desert", "minute",
    "discard", "rebel", "conduct", "contract", "contrast", "convict",
    "extract", "use", "does", "house", "separate"
}

CONTEXT_WORDS = {"the", "a", "an", "to", "that", "there", "in", "am", "i", "this"}


def classify_mismatch(text, ref_w, got_w):
    # check if the text contains a possessive near this word
    text_words = [w.strip(strip).lower() for w in text.split()]
    ref_low = ref_w.lower()
    got_low = got_w.lower() if got_w else ""

    # 1. the / a / to / that context
    if ref_w in ("ði", "ðə") and got_w in ("ði", "ðə"):
        return "the/a/to context"
    if ref_w in ("tə", "tu", "tʊ") and got_w in ("tə", "tu", "tʊ"):
        return "the/a/to context"
    if ref_w in ("ɐ", "ˈA", "ɐn") and got_w in ("ɐ", "ˈA", "ɐn"):
        return "the/a/to context"
    if ref_w in ("ðˈæt", "ðæt") and got_w in ("ðˈæt", "ðæt"):
        return "the/a/to context"
    if ref_w in ("ðˌɛɹ", "ðɛɹ", "ðˈɛɹ") and got_w in ("ðˌɛɹ", "ðɛɹ", "ðˈɛɹ"):
        return "the/a/to context"
    if ref_w in ("ɪn", "ˈɪn") and got_w in ("ɪn", "ˈɪn"):
        return "the/a/to context"

    # 2. Possessive ('s)
    has_poss = any(w.endswith("'s") or w.endswith("s'") or w == "'" for w in text_words)
    if has_poss and (ref_w.endswith("z") or ref_w.endswith("s") or not got_w):
        return "possessive"

    # 3. Heteronyms
    for h in HETERONYMS:
        if h in text_words:
            # likely heteronym cause
            return "heteronym"

    # 4. Compound
    # Check if a word in text is a known compound
    compounds = [
        "sunflower", "raincoat", "firefighter", "teaspoon", "stonemasons",
        "blackbird", "lighthouse", "wildfire", "doorstep", "nightfall",
        "bookstore", "notebook", "earthquake", "crossroads", "footstep",
        "waterfall", "snowstorm", "sandstorm", "barefoot", "shipwreck",
        "railroad", "keyboard", "seashell", "bedtime", "daydream",
        "headache", "toothache", "heartbeat", "fingerprint", "eyebrow",
        "sunlight", "moonlight", "downhill", "uphill", "outdoors", "indoors"
    ]
    if any(c in text_words for c in compounds):
        return "compound"

    # 5. Stress difference (same phone sequence without ˈ ˌ)
    strip_stress = lambda s: s.replace("ˈ", "").replace("ˌ", "")
    if strip_stress(ref_w) == strip_stress(got_w):
        return "stress / reduction"

    # 6. Other POS-aware
    if ref_w.startswith("ˈ") != got_w.startswith("ˈ"):
        return "POS-aware"

    return "other"


def main():
    exact = 0
    matched_words = 0
    total_err = 0
    oov_count = 0
    causes = Counter()
    sample_mismatches = []

    for r in dev_refs:
        text = r["text"]
        ref = r["phonemes"]
        p = subprocess.run([EXE, text], capture_output=True, timeout=30)
        got = p.stdout.decode("utf-8").rstrip("\n")
        err = p.stderr.decode("utf-8").strip()
        oov = 1 if err.startswith("warning: OOV") else 0
        oov_count += oov

        rw = [w.strip(strip) for w in ref.split() if w.strip(strip)]
        gw = [w.strip(strip) for w in got.split() if w.strip(strip)]

        if oov:
            total_err += len(ref)
        else:
            total_err += lev(got, ref)

        if not oov and got == ref:
            exact += 1
            matched_words += len(rw)
        else:
            sm = difflib.SequenceMatcher(a=gw, b=rw, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    matched_words += j2 - j1
                elif tag == "replace":
                    for k in range(min(j2 - j1, i2 - i1)):
                        rw_k = rw[j1 + k]
                        gw_k = gw[i1 + k]
                        cause = classify_mismatch(text, rw_k, gw_k)
                        causes[cause] += 1
                        if len(sample_mismatches) < 40 and cause != "the/a/to context":
                            sample_mismatches.append((cause, text[:50], rw_k, gw_k))
                elif tag == "delete":
                    # words in ref missing in got
                    for k in range(i2 - i1):
                        cause = classify_mismatch(text, "", gw[i1 + k])
                        causes[cause] += 1
                elif tag == "insert":
                    # words in ref missing from got (e.g. dropped OOV!)
                    for k in range(j2 - j1):
                        rw_k = rw[j1 + k]
                        cause = classify_mismatch(text, rw_k, "")
                        causes[cause] += 1
                        if len(sample_mismatches) < 40:
                            sample_mismatches.append((cause, text[:50], rw_k, "(missing)"))

    n = len(dev_refs)
    print("==================================================")
    print(f"DEV Set Baseline (current Rust HEAD, 525 sentences)")
    print("==================================================")
    print(f"Exact sentence match: {exact}/{n} = {100.0 * exact / n:.1f}%")
    print(f"Word-level phoneme match: {matched_words}/{TOTAL_WORDS} = {100.0 * matched_words / TOTAL_WORDS:.2f}%")
    print(f"PER (all, OOV = full error): {100.0 * total_err / TOTAL_CHARS:.2f}%")
    print(f"Sentences with OOV: {oov_count}/{n}")
    print()
    print("Breakdown of word mismatches by cause:")
    total_mismatches = sum(causes.values())
    for cause, cnt in causes.most_common():
        pct = 100.0 * cnt / max(1, total_mismatches)
        print(f"  {cause:25s}: {cnt:4d} ({pct:5.1f}%)")
    print(f"  {'TOTAL':25s}: {total_mismatches:4d}")
    print()
    print("Sample non-context mismatches (cause | text | ref <- got):")
    for cause, txt, rw_k, gw_k in sample_mismatches[:25]:
        print(f"  [{cause:15s}] {txt!r} : {rw_k!r} <- {gw_k!r}")


if __name__ == "__main__":
    main()
