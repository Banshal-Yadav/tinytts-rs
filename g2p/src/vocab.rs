//! Vocab mapping from IPA characters to model token IDs.
//!
//! Generated from repo root `vocab.txt`.

pub const VOCAB_TABLE: &[(char, i64)] = &[
    (' ', 16),
    ('!', 5),
    ('"', 11),
    ('(', 12),
    (')', 13),
    (',', 3),
    ('.', 4),
    (':', 2),
    (';', 1),
    ('?', 6),
    ('A', 24),
    ('I', 25),
    ('O', 31),
    ('Q', 33),
    ('S', 35),
    ('T', 36),
    ('W', 39),
    ('Y', 41),
    ('a', 43),
    ('b', 44),
    ('c', 45),
    ('d', 46),
    ('e', 47),
    ('f', 48),
    ('h', 50),
    ('i', 51),
    ('j', 52),
    ('k', 53),
    ('l', 54),
    ('m', 55),
    ('n', 56),
    ('o', 57),
    ('p', 58),
    ('q', 59),
    ('r', 60),
    ('s', 61),
    ('t', 62),
    ('u', 63),
    ('v', 64),
    ('w', 65),
    ('x', 66),
    ('y', 67),
    ('z', 68),
    ('æ', 72),
    ('ç', 78),
    ('ð', 81),
    ('ø', 116),
    ('ŋ', 112),
    ('œ', 120),
    ('ɐ', 70),
    ('ɑ', 69),
    ('ɒ', 71),
    ('ɔ', 76),
    ('ɕ', 77),
    ('ɖ', 80),
    ('ə', 83),
    ('ɚ', 85),
    ('ɛ', 86),
    ('ɜ', 87),
    ('ɟ', 90),
    ('ɡ', 92),
    ('ɣ', 139),
    ('ɤ', 140),
    ('ɥ', 99),
    ('ɨ', 101),
    ('ɪ', 102),
    ('ɯ', 110),
    ('ɰ', 111),
    ('ɲ', 114),
    ('ɳ', 113),
    ('ɴ', 115),
    ('ɸ', 118),
    ('ɹ', 123),
    ('ɻ', 126),
    ('ɽ', 129),
    ('ɾ', 125),
    ('ʁ', 128),
    ('ʂ', 130),
    ('ʃ', 131),
    ('ʈ', 132),
    ('ʊ', 135),
    ('ʋ', 136),
    ('ʌ', 138),
    ('ʎ', 143),
    ('ʒ', 147),
    ('ʔ', 148),
    ('ʝ', 103),
    ('ʣ', 18),
    ('ʤ', 82),
    ('ʥ', 19),
    ('ʦ', 20),
    ('ʧ', 133),
    ('ʨ', 21),
    ('ʰ', 162),
    ('ʲ', 164),
    ('ˈ', 156),
    ('ˌ', 157),
    ('ː', 158),
    ('̃', 17),
    ('β', 75),
    ('θ', 119),
    ('χ', 142),
    ('ᵊ', 42),
    ('ᵝ', 22),
    ('ᵻ', 177),
    ('—', 9),
    ('“', 14),
    ('”', 15),
    ('…', 10),
    ('→', 171),
    ('↓', 169),
    ('↗', 172),
    ('↘', 173),
    ('ꭧ', 23),
];

/// Look up token ID for a single character.
pub fn char_to_id(c: char) -> Option<i64> {
    VOCAB_TABLE
        .binary_search_by_key(&c, |&(ch, _)| ch)
        .ok()
        .map(|idx| VOCAB_TABLE[idx].1)
}

/// Maximum phonemes model accepts before BOS/EOS (512 total - 2).
pub const MAX_PHONEMES: usize = 510;

/// Map a phoneme string to sequence of token IDs, bounded by BOS (0) and EOS (0).
pub fn phonemes_to_ids(phonemes: &str) -> Vec<i64> {
    let mut ids = Vec::with_capacity(phonemes.len() + 2);
    ids.push(0);
    for c in phonemes.chars() {
        if let Some(id) = char_to_id(c) {
            if ids.len() <= MAX_PHONEMES {
                ids.push(id);
            }
        }
    }
    ids.push(0);
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_lookup() {
        assert_eq!(char_to_id(';'), Some(1));
        assert_eq!(char_to_id(' '), Some(16));
        assert_eq!(char_to_id('“'), Some(14));
        assert_eq!(char_to_id('”'), Some(15));
        assert_eq!(char_to_id('ᵻ'), Some(177));
        assert_eq!(char_to_id('@'), None);
    }

    #[test]
    fn test_phonemes_to_ids() {
        let ids = phonemes_to_ids("həlˈO");
        assert_eq!(ids.first(), Some(&0));
        assert_eq!(ids.last(), Some(&0));
    }
}
