# Held-out evaluation set (frozen)

Frozen on 2026-08-07 (Day 1 of G2P v1), branch `g2p-v1`. **Do not edit these
files afterwards.** All gate numbers are measured only on this set.

| File | SHA-256 | Size |
|---|---|---|
| `heldout.json` | `08dba05157e08ca2ac000b4cc3994d716d39f1d2293e9def8ef0a76f0b394f59` | 25,884 bytes |
| `heldout_ref.json` | `0c91fdddd239efab3947881f8c29400740608fe9e98d7faeec772229da226c3a` | 178,057 bytes |

## Contents

204 English sentences, none of which appear in the spike's 520-sentence
development set (`g2p-spike/sentences.json` / `ref.json`). Overlap was
verified programmatically: case-insensitive, whitespace-normalized exact
match against every text in both spike files — 0 matches.

| Source | Count |
|---|---|
| prose (common narration-style) | 74 |
| news-style | 40 |
| numbers (amounts, times, dates, years, counts) | 40 |
| dialogue (quoted speech) | 30 |
| mixed (numbers + dates + currency in prose) | 20 |

## Generation command

Reference phonemes were produced with upstream misaki via the repo's normal
code path (same as the spike's `get_ref.py`):

```
.venv\Scripts\python.exe scripts\get_heldout_ref.py
# internally: paradee.tts.Paradee(version="1.1", quantized=True,
#             model_path="paradee_int8.onnx").phonemize(text)
# ids = [0] + [vocab[c] for c in phonemes if c in vocab][:MAX_PHONEMES] + [0]
```

The set itself was written by `scripts/make_heldout.py`, which also performs
the overlap check and refuses to write on any overlap or duplicate.
