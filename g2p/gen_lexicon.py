"""Generate the compact lexicon binary embedded by the tinytts-g2p crate.

Sources (do not redistribute without keeping THIRD_PARTY_NOTICES.md):
  - misaki 0.9.4 us_gold.json / us_silver.json (Apache-2.0)
  - CMUdict cmusphinx/cmudict (BSD-style CMU license)

Output format (little-endian, all offsets u32):
  header:
    4s  magic "G2P1"
    u32 gold_count, silver_count, cmu_count
    u32 words_off, phonemes_off
  words section: per entry
    u32 byte offset into phonemes section
    u16 phoneme byte length
    u8  flags (bit 0: entry is a dict-valued DEFAULT resolved string)
    u8  word byte length
    [word bytes utf-8]
  phonemes section: concatenated utf-8 phoneme strings

Entries are sorted by word bytes so the crate can binary-search. Case
variants are pre-grown exactly like misaki's Lexicon.grow_dictionary so the
Rust side needs no case logic. Heteronym dict entries are resolved to their
DEFAULT form (v1 has no POS tagger; documented limitation).

Usage:
  python gen_lexicon.py <out.bin>
"""
import json
import os
import struct
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, ".venv", "Lib", "site-packages", "misaki", "data")
CMU = os.path.join(os.path.dirname(REPO), "g2p-spike", "cmudict.dict")
MAGIC = b"G2P1"


def grow_dictionary(d):
    """Same case-variant growth misaki's Lexicon.grow_dictionary does."""
    e = {}
    for k, v in d.items():
        if len(k) < 2:
            continue
        if k == k.lower() and k != k.capitalize():
            e[k.capitalize()] = v
        elif k == k.lower().capitalize():
            e[k.lower()] = v
    return {**e, **d}


def resolve(v):
    """Dict-valued (heteronym) entries resolve to DEFAULT for v1."""
    if isinstance(v, dict):
        return v.get("DEFAULT")
    return v


def load_misaki(path):
    raw = json.load(open(path, encoding="utf-8"))
    grown = grow_dictionary(raw)
    out = {}
    for k, v in grown.items():
        ps = resolve(v)
        if isinstance(ps, str) and ps:
            out[k] = ps
    return out


def load_cmu(path):
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or line.startswith(";;;"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word = parts[0].lower()
        # keep the first pronunciation; skip entries with digits in the word
        if word not in out and word.replace("'", "").isalpha():
            out[word] = " ".join(parts[1:])  # raw ARPAbet, mapped in Rust
    return out


def build(entries):
    """entries: list of (word, phonemes). Returns the binary blob."""
    entries = sorted(set(entries), key=lambda e: e[0].encode("utf-8"))
    words_blob = bytearray()
    phonemes_blob = bytearray()
    index = []
    for word, ps in entries:
        wb = word.encode("utf-8")
        pb = ps.encode("utf-8")
        index.append((len(phonemes_blob), len(pb), wb))
        phonemes_blob += pb
    for off, plen, wb in index:
        words_blob += struct.pack("<IHB", off, plen, len(wb)) + wb
    header = MAGIC + struct.pack(
        "<IIII", len(entries), 20, 20 + len(words_blob), 0
    )
    return header + words_blob + phonemes_blob


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "src", "lexicon.bin")
    gold = load_misaki(os.path.join(DATA, "us_gold.json"))
    silver = load_misaki(os.path.join(DATA, "us_silver.json"))
    cmu = load_cmu(CMU)
    # silver must not shadow gold: build gold first, add silver only if absent
    entries = [(w, ps) for w, ps in gold.items()]
    entries += [(w, ps) for w, ps in silver.items() if w not in gold]
    entries += [(w, ps) for w, ps in cmu.items()
                if w not in gold and w not in silver]
    blob = build(entries)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "wb").write(blob)
    print(f"gold={len(gold)} silver_new={len(silver) - len(gold)} "
          f"cmu_new={len(entries) - len(gold) - (len(silver) - len(gold))} "
          f"total={len(entries)}")
    print(f"wrote {out_path}: {len(blob)} bytes "
          f"({len(blob) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
