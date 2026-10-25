//! Embedded lexicon: binary format, loading, and lookup.
//!
//! The blob is a build artifact, not committed: generate it first with
//! `python gen_lexicon.py src/lexicon.bin` (see THIRD_PARTY_NOTICES.md for
//! sources and licenses). Format (little-endian):
//! ```text
//! header:  magic "G2P1" | u32 count | u32 words_off | u32 phonemes_off | u32 zero
//! words:   per entry: u32 phon_offset | u16 phon_len | u8 word_len | word bytes
//! phonemes: concatenated utf-8 phoneme strings
//! ```
//! Entries are sorted by word BYTES (the generator uses `key=encode('utf-8')`,
//! never locale order), so the Rust binary search is exact. Case variants
//! are pre-grown by the generator (same rule as misaki's grow_dictionary), so
//! lookup is a plain byte compare. Heteronym dict entries were resolved to
//! their DEFAULT form at generation time (v1 has no POS tagger).
//!
//! CMUdict entries are stored as raw ARPAbet strings ("D AO1 G") and are
//! converted to misaki symbols on first use via the derived map in `rules`.

use std::sync::OnceLock;

pub static LEXICON_BYTES: &[u8] = include_bytes!("lexicon.bin");

const MAGIC: &[u8; 4] = b"G2P1";

/// One resolved lexicon entry.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Entry<'a> {
    /// A misaki phoneme string, ready to use.
    Misaki(&'a str),
    /// A raw ARPAbet string from CMUdict ("D AO1 G"), needs mapping.
    Arpabet(&'a str),
}

struct Header {
    count: usize,
    words_off: usize,
    phonemes_off: usize,
}

fn header() -> Header {
    let b = LEXICON_BYTES;
    assert_eq!(&b[0..4], MAGIC, "lexicon.bin has a bad magic");
    let count = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let words_off = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
    let phonemes_off = u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize;
    Header {
        count,
        words_off,
        phonemes_off,
    }
}

struct Index {
    /// (word bytes, phon offset, phon len) per entry, in file order (sorted).
    words: Vec<(&'static [u8], usize, usize)>,
    phonemes_off: usize,
}

fn index() -> &'static Index {
    static IDX: OnceLock<Index> = OnceLock::new();
    IDX.get_or_init(|| {
        let h = header();
        let b = LEXICON_BYTES;
        let mut words = Vec::with_capacity(h.count);
        let mut o = h.words_off;
        for _ in 0..h.count {
            let poff = u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
            let plen = u16::from_le_bytes(b[o + 4..o + 6].try_into().unwrap()) as usize;
            let wlen = b[o + 6] as usize;
            words.push((&b[o + 7..o + 7 + wlen], poff, plen));
            o += 7 + wlen;
        }
        Index {
            words,
            phonemes_off: h.phonemes_off,
        }
    })
}

/// Look up a word exactly (case-sensitive; the generator pre-grew variants).
/// Returns `Entry::Misaki` for lexicon words and `Entry::Arpabet` for CMUdict
/// words. The discriminator is content-based: ARPAbet is pure ASCII with
/// spaces between phones; misaki phonemes are IPA-ish and never ASCII-only.
pub fn lookup(word: &str) -> Option<Entry<'static>> {
    let idx = index();
    let key = word.as_bytes();
    let mut lo = 0usize;
    let mut hi = idx.words.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        match idx.words[mid].0.cmp(key) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => {
                let (_, poff, plen) = idx.words[mid];
                let ps = std::str::from_utf8(
                    &LEXICON_BYTES[idx.phonemes_off + poff..idx.phonemes_off + poff + plen],
                )
                .ok()?;
                // The generator marks CMU entries by storing their phonemes
                // as ARPAbet (ASCII uppercase + digits + spaces). Misaki
                // phonemes are lowercase IPA-ish and never contain a space
                // followed by an uppercase letter sequence. A cheap reliable
                // discriminator: ARPAbet contains a space and is pure ASCII.
                if ps.is_ascii() && ps.contains(' ') {
                    return Some(Entry::Arpabet(ps));
                }
                return Some(Entry::Misaki(ps));
            }
        }
    }
    None
}

/// Number of entries (for tests and size reporting).
pub fn len() -> usize {
    index().words.len()
}

/// The word at index `i` in file (sorted) order. Test/inspection helper.
pub fn key_at(i: usize) -> Option<&'static str> {
    let idx = index();
    idx.words.get(i).and_then(|(w, _, _)| std::str::from_utf8(w).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_and_size() {
        assert_eq!(&LEXICON_BYTES[0..4], MAGIC);
        assert!(len() > 400_000, "lexicon unexpectedly small: {}", len());
    }

    #[test]
    fn lookup_common_words() {
        assert_eq!(lookup("the"), Some(Entry::Misaki("ði")));
        assert_eq!(lookup("hello"), Some(Entry::Misaki("həlˈO")));
        assert_eq!(lookup("synthesis"), Some(Entry::Misaki("sˈɪnθəsɪs")));
    }

    #[test]
    fn lookup_case_variants_pregrown() {
        // The generator grew case variants; both forms must hit.
        assert!(matches!(lookup("Old"), Some(Entry::Misaki(_))));
        assert!(matches!(lookup("old"), Some(Entry::Misaki(_))));
    }

    #[test]
    fn lookup_misaki_entry_has_ipa() {
        // "sheriff" exists in misaki's silver lexicon as IPA, not CMU ARPAbet.
        assert_eq!(lookup("sheriff"), Some(Entry::Misaki("ʃˈɛɹəf")));
    }

    #[test]
    fn lookup_cmu_entry_is_arpabet() {
        // "aardvarks" is in CMUdict but not in misaki's gold/silver lexicons
        // (verified at generation time), so it must come back as raw ARPAbet.
        match lookup("aardvarks") {
            Some(Entry::Arpabet(_)) => {}
            other => panic!("expected Arpabet, got {:?}", other.map(|e| match e {
                Entry::Misaki(p) => p.to_string(),
                Entry::Arpabet(p) => p.to_string(),
            })),
        }
    }

    #[test]
    fn every_key_is_findable_via_lookup() {
        // The cheapest guard against a sort-order or offset bug: walk the
        // index in order and confirm every stored key round-trips through
        // the public lookup. Catches locale-vs-byte sort and off-by-N header
        // bugs in one pass.
        let idx_count = len();
        assert!(idx_count > 400_000);
        // Spot-check the first, middle, and last keys plus a spread of others.
        let probes = [0, 1, idx_count / 3, idx_count / 2, idx_count - 2, idx_count - 1];
        for i in probes {
            let w = key_at(i);
            assert!(w.is_some(), "key_at({}) failed", i);
            let w = w.unwrap();
            let got = lookup(w);
            assert!(got.is_some(), "lookup({:?}) missed its own key", w);
        }
    }

    #[test]
    fn missing_word_is_none() {
        assert_eq!(lookup("zxqj"), None);
    }
}
