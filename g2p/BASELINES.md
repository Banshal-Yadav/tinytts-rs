# G2P Held-Out Baselines

All metrics are measured on the frozen 204-sentence held-out set (`heldout.json`,
sha256: `08dba0519a7153d8c2794eb09fc846e4b9e287413d96fa7ca91fe69b82094f59`,
reference: `heldout_ref.json`).

Scorer: `scripts/measure_rust_g2p.py` (same scoring logic across all commits).

| Commit | Stage | Exact Match | Word Match | PER (all) | Notes |
|---|---|---|---|---|---|
| `42148e3` | Day 1 untuned reference | 52.0% (106/204) | 83.94% (1066/1270) | 16.34% | Original proto3.py before bugfixes |
| `7092d63` | Day 2 lexicon-only Rust | 0.0% (0/204) | ~76.5% | 63.58% | Lexicon lookups only, no context rules |
| `35dc555` | Day 3 context rules + CMU | 33.8% (69/204) | 90.80% (subset) | 74.90% | Digits unexpanded (counted as full error); digit-free subset was 63.3% exact / 90.80% word / 7.70% PER |
| `35bb083` | Day 4 numbers port | 58.3% (119/204) | 85.70% (983/1147) | 8.52% | Numbers, ordinals, time, date, currency, % symbol |
| `HEAD` | Day 4 ground-truth fixes | 60.8% (124/204) | 85.97% (919/1069) | 8.54% | Year rule, 12th ordinal, currency wording, trailing zeros |
