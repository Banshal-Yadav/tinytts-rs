//! Minimal Day-2 rules module: tokenization, lexicon lookup with the
//! misaki/CMU split, punctuation gluing, and hyphenated-compound joining.
//! The CMU ARPAbet mapping and the full context rules land on Day 3; for now
//! an ARPAbet-only word is passed through as-is and reported as OOV-adjacent
//! (still counted, never silent).

use crate::lexicon::{self, Entry};
use crate::G2pResult;

/// Token classes produced by `tokenize`.
enum Tok {
    Word(String),
    Number(String),
    Punct(char),
    Other(char),
}

fn is_punct(c: char) -> bool {
    matches!(
        c,
        ';' | ':'
            | ','
            | '.'
            | '!'
            | '?'
            | '\u{2014}'
            | '\u{2026}'
            | '"'
            | '\u{201c}'
            | '\u{201d}'
            | '('
            | ')'
    )
}

fn tokenize(text: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_alphabetic() || c == '\'' && out.last().is_some() {
            // word: letters plus internal apostrophes ("That's")
            let mut w = String::new();
            while let Some(&c2) = chars.peek() {
                if c2.is_ascii_alphabetic() || (c2 == '\'' && !w.is_empty()) {
                    w.push(c2);
                    chars.next();
                } else {
                    break;
                }
            }
            // trim a trailing lone apostrophe ("dogs'" -> "dogs")
            while w.ends_with('\'') {
                w.pop();
            }
            if !w.is_empty() {
                out.push(Tok::Word(w));
            }
        } else if c.is_ascii_digit() {
            let mut n = String::new();
            while let Some(&c2) = chars.peek() {
                if c2.is_ascii_digit() || c2 == ',' || c2 == '.' || c2 == ':' {
                    n.push(c2);
                    chars.next();
                } else {
                    break;
                }
            }
            out.push(Tok::Number(n));
        } else if is_punct(c) {
            chars.next();
            out.push(Tok::Punct(c));
        } else if c.is_whitespace() {
            chars.next();
        } else {
            chars.next();
            out.push(Tok::Other(c));
        }
    }
    out
}

/// Look up one word: misaki entry first, CMUdict second.
fn word_phonemes(word: &str) -> Option<String> {
    match lexicon::lookup(word) {
        Some(Entry::Misaki(ps)) => Some(ps.to_string()),
        Some(Entry::Arpabet(_)) => None, // Day 3: ARPAbet mapping
        None => None,
    }
}

/// Convert text to phonemes. Day-2 scope: lexicon words + punctuation.
pub fn g2p(text: &str) -> G2pResult {
    let toks = tokenize(text);
    let mut parts: Vec<String> = Vec::new();
    let mut oov = Vec::new();
    for tok in &toks {
        match tok {
            Tok::Punct(p) => {
                // glue to the previous chunk: no space before punctuation
                if let Some(last) = parts.last_mut() {
                    last.push(*p);
                } else {
                    parts.push(p.to_string());
                }
            }
            Tok::Word(w) => match word_phonemes(w) {
                Some(ps) => parts.push(ps),
                None => {
                    oov.push(w.clone());
                    parts.push(w.clone()); // visible placeholder, never silent
                }
            },
            Tok::Number(n) => {
                // Day 4: number expansion. Until then the digits are a
                // reported gap, not a silent one.
                oov.push(n.clone());
                parts.push(n.clone());
            }
            Tok::Other(c) => parts.push(c.to_string()),
        }
    }
    // join with spaces, but no space before punctuation (already glued above)
    let mut out = String::new();
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(p);
    }
    G2pResult { phonemes: out, oov }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexicon_sentence() {
        let r = g2p("Hello there.");
        assert_eq!(r.phonemes, "həlˈO ðɛɹ.");
        assert!(r.oov.is_empty());
    }

    #[test]
    fn punctuation_glues() {
        let r = g2p("Wait, really?");
        assert!(r.phonemes.starts_with("wˈAt, ɹˈiᵊli?"), "{}", r.phonemes);
    }

    #[test]
    fn oov_is_reported_not_silent() {
        let r = g2p("The zxqj sat.");
        assert_eq!(r.oov, vec!["zxqj".to_string()]);
        assert!(r.phonemes.contains("zxqj"));
    }

    #[test]
    fn numbers_are_reported_until_day_4() {
        let r = g2p("I have 3 cats.");
        assert_eq!(r.oov, vec!["3".to_string()]);
    }
}
