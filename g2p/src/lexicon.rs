//! Embedded lexicon: binary format, loading, and lookup.
//!
//! The blob is a build artifact, not committed: generate it first with
//! `python gen_lexicon.py src/lexicon.bin` (see THIRD_PARTY_NOTICES.md for
//! sources and licenses). Format v2 (little-endian):
//! ```text
//! header (28 bytes):
//!   magic "G2P2" | u32 misaki_count | u32 cmu_count
//!   u32 misaki_words_off | u32 misaki_phonemes_off
//!   u32 cmu_words_off    | u32 cmu_phonemes_off
//! per entry (both sections):
//!   u32 phon_offset (relative to that section's phoneme blob)
//!   u16 phon_len | u8 word_len | word bytes
//! phoneme blobs: concatenated utf-8 strings
//! ```
//! The misaki section holds gold+silver (gold wins) with case variants
//! pre-grown by the generator, so lookup there is a plain byte compare.
//! The CMU section holds the FULL cmudict (lowercase keys, raw ARPAbet) -
//! the same view the Python reference uses, so the stem rules in `rules`
//! see exactly the same stems.
//!
//! Entries are sorted by word BYTES (the generator uses `key=encode('utf-8')`,
//! never locale order), so the Rust binary search is exact.

use std::sync::OnceLock;

pub static LEXICON_BYTES: &[u8] = include_bytes!("lexicon.bin");

const MAGIC: &[u8; 4] = b"G2P2";

/// Which lexicon section to inspect (test helper selector).
#[derive(Clone, Copy, Debug)]
pub enum Section {
    Misaki,
    Cmu,
}

/// One resolved lexicon entry.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Entry<'a> {
    /// A misaki phoneme string, ready to use.
    Misaki(&'a str),
    /// A raw ARPAbet string from CMUdict ("D AO1 G"), needs mapping.
    Arpabet(&'a str),
}

struct SectionData {
    /// (word bytes, phon offset, phon len) per entry, in file order (sorted).
    words: Vec<(&'static [u8], usize, usize)>,
    phonemes_off: usize,
}

struct Index {
    misaki: SectionData,
    cmu: SectionData,
}

fn u32le(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize
}

fn parse_section(
    b: &'static [u8],
    words_off: usize,
    phonemes_off: usize,
    count: usize,
) -> SectionData {
    let mut words = Vec::with_capacity(count);
    let mut o = words_off;
    for _ in 0..count {
        let poff = u32le(b, o);
        let plen = u16::from_le_bytes(b[o + 4..o + 6].try_into().unwrap()) as usize;
        let wlen = b[o + 6] as usize;
        words.push((&b[o + 7..o + 7 + wlen], poff, plen));
        o += 7 + wlen;
    }
    SectionData {
        words,
        phonemes_off,
    }
}

fn index() -> &'static Index {
    static IDX: OnceLock<Index> = OnceLock::new();
    IDX.get_or_init(|| {
        let b = LEXICON_BYTES;
        assert_eq!(&b[0..4], MAGIC, "lexicon.bin has a bad magic");
        let misaki_count = u32le(b, 4);
        let cmu_count = u32le(b, 8);
        let misaki_words_off = u32le(b, 12);
        let misaki_phonemes_off = u32le(b, 16);
        let cmu_words_off = u32le(b, 20);
        let cmu_phonemes_off = u32le(b, 24);
        Index {
            misaki: parse_section(b, misaki_words_off, misaki_phonemes_off, misaki_count),
            cmu: parse_section(b, cmu_words_off, cmu_phonemes_off, cmu_count),
        }
    })
}

fn phon_str(sec: &SectionData, poff: usize, plen: usize) -> Option<&'static str> {
    std::str::from_utf8(
        &LEXICON_BYTES[sec.phonemes_off + poff..sec.phonemes_off + poff + plen],
    )
    .ok()
}

fn binary_search(sec: &SectionData, key: &[u8]) -> Option<(&'static [u8], usize, usize)> {
    let mut lo = 0usize;
    let mut hi = sec.words.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        match sec.words[mid].0.cmp(key) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Some(sec.words[mid]),
        }
    }
    None
}

/// Look up a word in the misaki section (case-sensitive; the generator
/// pre-grew variants). Returns the ready-to-use phoneme string.
pub fn lookup_misaki(word: &str) -> Option<&'static str> {
    let idx = index();
    let (w, poff, plen) = binary_search(&idx.misaki, word.as_bytes())?;
    debug_assert_eq!(w, word.as_bytes());
    phon_str(&idx.misaki, poff, plen)
}

/// Look up a word in the full CMU section (lowercase keys). Returns the raw
/// ARPAbet string ("D AO1 G") for the `rules` module to map.
pub fn lookup_cmu(word: &str) -> Option<&'static str> {
    let idx = index();
    let (w, poff, plen) = binary_search(&idx.cmu, word.as_bytes())?;
    debug_assert_eq!(w, word.as_bytes());
    phon_str(&idx.cmu, poff, plen)
}

/// Combined lookup: misaki first, CMU second. Kept for callers that do not
/// care which section answered.
pub fn lookup(word: &str) -> Option<Entry<'static>> {
    if let Some(ps) = lookup_misaki(word) {
        return Some(Entry::Misaki(ps));
    }
    lookup_cmu(word).map(Entry::Arpabet)
}

/// Number of entries in the misaki section (for tests and size reporting).
pub fn misaki_len() -> usize {
    index().misaki.words.len()
}

/// Number of entries in the CMU section.
pub fn cmu_len() -> usize {
    index().cmu.words.len()
}

/// The word at index `i` in file (sorted) order of the given section.
/// Test/inspection helper.
pub fn key_at(section: Section, i: usize) -> Option<&'static str> {
    let idx = index();
    let sec = match section {
        Section::Misaki => &idx.misaki,
        Section::Cmu => &idx.cmu,
    };
    sec.words
        .get(i)
        .and_then(|(w, _, _)| std::str::from_utf8(w).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_and_sizes() {
        assert_eq!(&LEXICON_BYTES[0..4], MAGIC);
        assert!(misaki_len() > 300_000, "misaki section small: {}", misaki_len());
        assert!(cmu_len() > 130_000, "cmu section small: {}", cmu_len());
    }

    #[test]
    fn lookup_common_words() {
        assert_eq!(lookup_misaki("the"), Some("ði"));
        assert_eq!(lookup_misaki("hello"), Some("həlˈO"));
        assert_eq!(lookup_misaki("synthesis"), Some("sˈɪnθəsɪs"));
    }

    #[test]
    fn lookup_case_variants_pregrown() {
        // The generator grew case variants; both forms must hit.
        assert!(lookup_misaki("Old").is_some());
        assert!(lookup_misaki("old").is_some());
    }

    #[test]
    fn cmu_section_is_full_dict() {
        // "cat" is in BOTH misaki and cmudict. The misaki section answers
        // first for the combined lookup, but the CMU section must still
        // contain the raw ARPAbet so the stem rules see the same stems as
        // the Python reference.
        assert_eq!(lookup_cmu("cat"), Some("K AE1 T"));
        // "aardvarks" is CMU-only.
        assert!(lookup_misaki("aardvarks").is_none());
        assert_eq!(lookup_cmu("aardvarks"), Some("AA1 R D V AA2 R K S"));
        // combined lookup prefers misaki
        assert!(matches!(lookup("cat"), Some(Entry::Misaki(_))));
        assert!(matches!(lookup("aardvarks"), Some(Entry::Arpabet(_))));
    }

    #[test]
    fn every_key_is_findable_via_lookup() {
        // The cheapest guard against a sort-order or offset bug: walk the
        // index in order and confirm every stored key round-trips through
        // the public lookup. Catches locale-vs-byte sort and off-by-N header
        // bugs in one pass.
        for (section, count) in
            [(Section::Misaki, misaki_len()), (Section::Cmu, cmu_len())]
        {
            let probes = [0, 1, count / 3, count / 2, count - 2, count - 1];
            for i in probes {
                let w = key_at(section, i);
                assert!(w.is_some(), "key_at({:?}, {}) failed", section, i);
                let w = w.unwrap();
                let got = match section {
                    Section::Misaki => lookup_misaki(w).map(|_| ()),
                    Section::Cmu => lookup_cmu(w).map(|_| ()),
                };
                assert!(got.is_some(), "lookup({:?}) missed its own key", w);
            }
        }
    }

    #[test]
    fn missing_word_is_none() {
        assert_eq!(lookup("zxqj"), None);
        assert_eq!(lookup_cmu("zxqj"), None);
    }
}
