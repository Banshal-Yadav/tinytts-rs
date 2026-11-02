# G2P Held-Out Baselines

All metrics are measured on the frozen 204-sentence held-out set (`heldout.json`,
sha256: `08dba0519a7153d8c2794eb09fc846e4b9e287413d96fa7ca91fe69b82094f59`,
reference: `heldout_ref.json`).

Fixed Scorer Specification:
- **Exact Match**: number of sentences where `got == ref` and no OOV warning, divided by 204.
- **Word Match**: reference words across all 204 sentences (`heldout_ref.json`), split by
  whitespace and stripped of punctuation (`.,!?;:"“”‘’…—`). Denominator is fixed at **2,658 words**
  independent of model output. Numerator is the count of exact matching words aligned via
  `SequenceMatcher` (all words count for exact sentences).
- **PER (all)**: standard Phone Error Rate: total Levenshtein distance across all 204 sentences
  (OOV sentences count as full sentence length error) divided by total reference characters (15,172).

| Commit | Stage | Exact Match | Word Match (fixed denom) | PER (all) | OOV Sents | Notes |
|---|---|---|---|---|---|---|
| `42148e3` | Day 1 untuned reference | 106/204 (52.0%) | 2526/2658 (95.03%) | 8.03% | 7 | Original proto3.py before bugfixes |
| `7092d63` | Day 2 lexicon-only Rust | 0/204 (0.0%) | 1775/2658 (66.78%) | 64.39% | 122 | Lexicon lookups only, no context/numbers |
| `35dc555` | Day 3 context rules + CMU | 69/204 (33.8%) | 2526/2658 (95.03%) | 50.24% | 96 | Digits unexpanded (OOV full error) |
| `35bb083` | Day 4 numbers port | 106/204 (52.0%) | 2526/2658 (95.03%) | 8.03% | 7 | Numbers, ordinals, time, date, currency |
| `de75d85` | Day 4 ground-truth fixes | 124/204 (60.8%) | 2574/2658 (96.84%) | 3.33% | 2 | Year rule, 12th ordinal, currency, trailing zeros |
| `de75d85` | Day 5 context rules + OOV | 136/204 (66.7%) | 2587/2658 (97.33%) | 2.22% | 0 | Possessives, compound split, L2S, to/there/in/used |
