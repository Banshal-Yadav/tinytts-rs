"""Ground-truth check: fixed proto3 vs misaki (the real reference).

misaki is the upstream G2P the model was trained with; its number handling
goes through the `num2words` library. This script runs the same generated
sweep through both and reports every disagreement with a verdict.

To isolate NUMBER behaviour from context-rule differences (which are a
separate workstream), each input is a bare number token where possible.

Usage (from the repo root):
  python scripts/ground_truth_g2p.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(REPO, "g2p", "reference")

sys.path.insert(0, REFERENCE)
os.chdir(REFERENCE)
import proto3  # noqa: E402

from misaki import en  # noqa: E402

g2p = en.G2P(trf=False, british=False, fallback=None)

# bare number tokens: no surrounding words, so any diff is the number layer
inputs = []
for y in range(1000, 2101):
    inputs.append(str(y))
for n in range(1, 101):
    suf = "th"
    if n % 100 not in (11, 12, 13):
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    inputs.append(f"{n}{suf}")
for w in ["0.5", "3.14", "42.50", "100.00", "1,284", "12,000", "1,284.67",
          "45,000", "999,999", "1,000,000", "7.2", "3.8", "4.0", "0.001"]:
    inputs.append(w)
for w in ["$18", "$4.7", "$1,284.67", "$0.99", "$100", "$45,000"]:
    inputs.append(w)
for w in ["9:30", "6:45", "12:00", "3:05", "11:59", "7:22", "2:25", "6:07"]:
    inputs.append(w)
for w in ["12/3/1905", "6/19/1924", "11/8/1998", "1/1/2000", "7/4/1776"]:
    inputs.append(w)
for w in ["-7", "-42", "-1,000"]:
    inputs.append(w)

MISAKI_UNHANDLED = "\u2753"  # ❓

# Intentional divergences (documented in g2p/DIVERGENCES.md): we keep the
# better behaviour instead of copying misaki.
INTENTIONAL = {
    "-7", "-42", "-1,000",  # bare negatives: we say "minus seven"
    "9:30", "6:45", "12:00", "3:05", "11:59", "7:22", "2:25", "6:07",  # H:MM
}

disagreements = []
for t in inputs:
    pp, _ = proto3.g2p(t)
    mp, _ = g2p(t)
    if pp == mp:
        continue
    disagreements.append((t, pp, mp))

print(f"sweep inputs: {len(inputs)}")
print(f"disagreements (proto3 vs misaki): {len(disagreements)}\n")

for t, pp, mp in disagreements:
    if t in INTENTIONAL:
        verdict = "intentional divergence"
    elif MISAKI_UNHANDLED in mp:
        verdict = "misaki limitation"
    else:
        verdict = "proto3 bug"
    print(f"input: {t!r}")
    print(f"  proto3: {pp!r}")
    print(f"  misaki: {mp!r}")
    print(f"  verdict: {verdict}")
    print()

bugs = sum(1 for t, _, mp in disagreements
           if t not in INTENTIONAL and MISAKI_UNHANDLED not in mp)
lims = sum(1 for t, _, mp in disagreements
           if t not in INTENTIONAL and MISAKI_UNHANDLED in mp)
intent = sum(1 for t, _, _ in disagreements if t in INTENTIONAL)
print(f"proto3 bugs: {bugs}")
print(f"misaki limitations: {lims}")
print(f"intentional divergences: {intent}")
