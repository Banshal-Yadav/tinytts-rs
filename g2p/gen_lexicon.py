"""Generate the compact lexicon binary embedded by the tinytts-g2p crate.

Sources (do not redistribute without keeping THIRD_PARTY_NOTICES.md):
  - misaki 0.9.4 us_gold.json / us_silver.json (Apache-2.0)
  - CMUdict cmusphinx/cmudict (BSD-style CMU license)

Output format v2 (little-endian):
  header (28 bytes):
    4s  magic "G2P2"
    u32 misaki_count, cmu_count
    u32 misaki_words_off, misaki_phonemes_off
    u32 cmu_words_off, cmu_phonemes_off
  per entry (both sections):
    u32 phoneme offset (relative to that section's phoneme blob)
    u16 phoneme byte length
    u8  word byte length
    [word bytes utf-8]
  phoneme blobs: concatenated utf-8 strings

Misaki section: gold + silver (gold wins), case variants pre-grown exactly
like misaki's Lexicon.grow_dictionary, heteronym dicts resolved to DEFAULT.
CMU section: the FULL cmudict (first pronunciation per word, lowercased
keys, raw ARPAbet, comment lines skipped) - the same view of CMUdict the
Python reference (g2p-spike/proto3.py) uses, so the Rust stem rules
(-s/-ed/-ing) see exactly the same stems, including stems that are also in
the misaki lexicon.

Entries are sorted by word bytes so the crate can binary-search. Case
variants are pre-grown for the misaki section so the Rust side needs no
case logic there; the CMU section is all-lowercase by construction.

Usage:
  python gen_lexicon.py <out.bin>
"""
import json
import os
import struct
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def misaki_data_dir():
    """Locate misaki's data dir. Prerequisite: pip install misaki==0.9.4.
    Override with MISAKI_DATA_DIR if misaki is not importable."""
    env = os.environ.get("MISAKI_DATA_DIR")
    if env:
        return env
    import misaki
    return os.path.join(os.path.dirname(misaki.__file__), "data")


# cmudict.dict lives next to this script (gitignored, 3.4 MB, re-downloadable):
#   https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
CMU = os.path.join(REPO, "g2p", "cmudict.dict")
MAGIC = b"G2P2"


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
    """Same view of cmudict as the Python reference: first pronunciation per
    lowercased word, comment/blank lines skipped, everything else kept."""
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or line.startswith(";;;"):
            continue
        parts = line.split()
        if parts[0].lower() not in out:
            out[parts[0].lower()] = " ".join(parts[1:])
    return out


def build_section(entries):
    """entries: list of (word, phonemes). Returns (words_blob, phonemes_blob)."""
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
    return bytes(words_blob), bytes(phonemes_blob)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "src", "lexicon.bin")
    d = misaki_data_dir()
    gold = load_misaki(os.path.join(d, "us_gold.json"))
    silver = load_misaki(os.path.join(d, "us_silver.json"))
    cmu = load_cmu(CMU)
    # silver must not shadow gold: build gold first, add silver only if absent
    misaki_entries = [(w, ps) for w, ps in gold.items()]
    misaki_entries += [(w, ps) for w, ps in silver.items() if w not in gold]
    cmu_entries = [(w, ps) for w, ps in cmu.items()]
    mw, mp = build_section(misaki_entries)
    cw, cp = build_section(cmu_entries)
    header = MAGIC + struct.pack(
        "<IIIIII",
        len(misaki_entries), len(cmu_entries),
        28, 28 + len(mw), 28 + len(mw) + len(mp), 28 + len(mw) + len(mp) + len(cw),
    )
    blob = header + mw + mp + cw + cp
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "wb").write(blob)
    print(f"misaki={len(misaki_entries)} cmu={len(cmu_entries)} "
          f"total={len(misaki_entries) + len(cmu_entries)}")
    print(f"wrote {out_path}: {len(blob)} bytes "
          f"({len(blob) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
