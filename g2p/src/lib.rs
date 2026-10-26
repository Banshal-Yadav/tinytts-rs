//! tinytts-g2p: grapheme-to-phoneme conversion for tinytts.
//!
//! Converts English text to the misaki phoneme strings the Paradee-8M model
//! consumes. Zero dependencies: the lexicon is embedded static data and every
//! lookup is std-only.
//!
//! v1 pipeline (ported from the validated Python prototype):
//! 1. tokenize (words, numbers, punctuation)
//! 2. lexicon lookup: misaki gold, then silver, then CMUdict with the
//!    derived ARPAbet-to-misaki map
//! 3. misaki-style stress and reduction rules
//! 4. number/date/currency expansion
//! 5. OOV policy: never fail silently - every OOV word is reported and gets
//!    a letter-to-sound attempt
//!
//! Known limitation (documented, by design): no POS tagging, so heteronyms
//! ("lead", "read", "bass") always get their DEFAULT (noun) pronunciation.

pub mod arpabet;
pub mod lexicon;
pub mod numbers;
pub mod rules;

/// Result of converting one text to phonemes.
pub struct G2pResult {
    /// The misaki phoneme string, ready for the model's vocab mapping.
    pub phonemes: String,
    /// Words that were not found in any lexicon and fell through to
    /// letter-to-sound. Never silently empty when non-empty: these are the
    /// words whose pronunciation is least reliable.
    pub oov: Vec<String>,
}

/// Convert English text to misaki phonemes.
pub fn g2p(text: &str) -> G2pResult {
    rules::g2p(text)
}
