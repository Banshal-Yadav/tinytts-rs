//! Day-3 word pipeline: tokenization, misaki lookup, CMU ARPAbet fallback
//! with -s/-ed/-ing stem rules, function-word context rules, and the
//! misaki-style post passes (flap, flap_all, demote_later, cap_stress).
//! Ported line-for-line from the validated Python prototype
//! (g2p-spike/proto3.py); the differential test in `tests/diff.rs` enforces
//! byte-exact agreement with that reference on the spike corpus.
//!
//! Day-4 scope (numbers, dates, currency) is intentionally NOT here yet:
//! number tokens are reported as OOV-adjacent gaps, never silently dropped.

use crate::arpabet;
use crate::lexicon;
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
            // word: letters plus internal apostrophes ("That's"). A leading
            // quote ('Wait,') enters this branch with an empty word; the
            // quote must be consumed here or the outer loop spins forever.
            let mut w = String::new();
            if c == '\'' {
                chars.next();
            }
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

// ---------------------------------------------------------------------------
// CMU fallback with stem rules (proto3 cmu_lookup / add_s / add_ed / add_ing)
// ---------------------------------------------------------------------------

/// Split "AO1" into ("AO", "1").
fn split_stress(ph: &str) -> (&str, &str) {
    let base = ph.trim_end_matches(|c| ('0'..='2').contains(&c));
    (base, &ph[base.len()..])
}

fn add_s(stem: &str) -> Option<String> {
    let phones: Vec<&str> = stem.split(' ').collect();
    let (last_base, _last_stress) = split_stress(phones.last()?);
    let mut out = stem.to_string();
    if matches!(last_base, "S" | "Z" | "SH" | "ZH" | "CH" | "JH") {
        out.push_str(" IH0 Z");
    } else {
        out.push_str(" Z");
    }
    Some(out)
}

fn add_ed(stem: &str) -> Option<String> {
    let phones: Vec<&str> = stem.split(' ').collect();
    let (last_base, _last_stress) = split_stress(phones.last()?);
    let mut out = stem.to_string();
    if matches!(last_base, "T" | "D") {
        out.push_str(" IH0 D");
    } else if matches!(last_base, "P" | "K" | "F" | "TH" | "SH" | "S" | "CH") {
        out.push_str(" T");
    } else {
        out.push_str(" D");
    }
    Some(out)
}

fn add_ing(stem: &str) -> Option<String> {
    let mut out = stem.to_string();
    out.push_str(" IH0 NG");
    Some(out)
}

/// CMUdict lookup with the proto3 stem rules. `word` must be lowercase.
fn cmu_lookup(word: &str) -> Option<String> {
    let mut phones = lexicon::lookup_cmu(word).map(|s| s.to_string());
    if phones.is_none() {
        // -s: cats -> cat, boxes -> box (via -es), cities -> city
        if word.len() > 2 && word.ends_with('s') && !word.ends_with("ss") {
            let mut stems: Vec<String> = vec![word[..word.len() - 1].to_string()];
            if word.ends_with("es") {
                stems.push(word[..word.len() - 2].to_string());
            }
            if word.ends_with("ies") {
                stems.push(format!("{}y", &word[..word.len() - 3]));
            }
            for stem in &stems {
                if let Some(p) = lexicon::lookup_cmu(stem) {
                    phones = add_s(p);
                    break;
                }
            }
        }
    }
    if phones.is_none() {
        // -ed: walked -> walk, stopped -> stop
        if word.len() > 3 && word.ends_with('d') {
            let mut stems: Vec<String> = vec![word[..word.len() - 1].to_string()];
            if word.ends_with("ed") && !word.ends_with("eed") {
                stems.push(word[..word.len() - 2].to_string());
            }
            for stem in &stems {
                if let Some(p) = lexicon::lookup_cmu(stem) {
                    phones = add_ed(p);
                    break;
                }
            }
        }
    }
    if phones.is_none() {
        // -ing: walking -> walk, making -> mak(e), running -> run(n)
        if word.len() > 4 && word.ends_with("ing") {
            let stem1 = word[..word.len() - 3].to_string();
            let stem2 = format!("{}e", stem1);
            // proto3 regex: ([bcdgklmnprstvxz])\1ing$ or cking$
            let b = word.as_bytes();
            let doubled = b.len() >= 5
                && b[b.len() - 5] == b[b.len() - 4]
                && matches!(
                    b[b.len() - 5],
                    b'b' | b'c' | b'd' | b'g' | b'k' | b'l' | b'm' | b'n' | b'p' | b'r' | b's'
                        | b't' | b'v' | b'x' | b'z'
                );
            let mut stems: Vec<String> = vec![stem1, stem2];
            if doubled {
                stems.push(word[..word.len() - 4].to_string());
            }
            for stem in &stems {
                if let Some(p) = lexicon::lookup_cmu(stem) {
                    phones = add_ing(p);
                    break;
                }
            }
        }
    }
    let phones = phones?;
    let parts: Vec<&str> = phones.split(' ').collect();
    let ps = arpabet::convert(&parts)?;
    Some(arpabet::demote_later(&arpabet::flap_all(&arpabet::flap(&ps))))
}

/// misaki gold+silver lookup (the pre-grown case variants make this a plain
/// byte compare, exactly like proto3's GOLD.get(word) or GOLD.get(lower)).
fn lexicon_lookup(word: &str) -> Option<&'static str> {
    lexicon::lookup_misaki(word).or_else(|| {
        let low = word.to_lowercase();
        if low == word {
            None
        } else {
            lexicon::lookup_misaki(&low)
        }
    })
}

// ---------------------------------------------------------------------------
// Function-word context rules (proto3 g2p, word-level branch)
// ---------------------------------------------------------------------------

/// Letter-by-letter spelling using misaki's gold letter names (proto3
/// LETTERS). Returns letters joined with spaces; the acronym path strips
/// the spaces.
fn spell_out(word: &str) -> Option<String> {
    const LETTERS: &[(&str, &str)] = &[
        ("A", "ˈA"), ("B", "bˈi"), ("C", "sˈi"), ("D", "dˈi"), ("E", "ˈi"),
        ("F", "ˈɛf"), ("G", "ʤˈi"), ("H", "ˈAʧ"), ("I", "ˈI"), ("J", "ʤˈA"),
        ("K", "kˈA"), ("L", "ˈɛl"), ("M", "ˈɛm"), ("N", "ˈɛn"), ("O", "ˈO"),
        ("P", "pˈi"), ("Q", "kjˈu"), ("R", "ˈɑɹ"), ("S", "ˈɛs"), ("T", "tˈi"),
        ("U", "jˈu"), ("V", "vˈi"), ("W", "dˈʌbᵊlju"), ("X", "ˈɛks"),
        ("Y", "wˈI"), ("Z", "zˈi"),
    ];
    let mut out = Vec::new();
    for c in word.chars() {
        if !c.is_ascii_alphabetic() {
            return None;
        }
        let up = c.to_ascii_uppercase().to_string();
        let ps = LETTERS.iter().find(|(k, _)| *k == up).map(|(_, v)| *v)?;
        out.push(ps);
    }
    if out.is_empty() {
        None
    } else {
        Some(out.join(" "))
    }
}

/// Phonemes of the next word-ish token (for the/a sound checks).
fn next_word_phonemes(idx: usize, toks: &[Tok]) -> Option<String> {
    for j in idx + 1..(idx + 3).min(toks.len()) {
        match &toks[j] {
            Tok::Punct(_) => continue,
            Tok::Other(c) if *c == '$' || *c == '-' || *c == '\u{2014}' => continue,
            Tok::Word(w) => {
                if w == "a" || w == "A" {
                    return Some("ɐ".to_string());
                }
                if w == "an" {
                    return Some("ɐn".to_string());
                }
                if let Some(ps) = lexicon_lookup(w) {
                    return Some(ps.to_string());
                }
                if let Some(ps) = cmu_lookup(&w.to_lowercase()) {
                    return Some(ps);
                }
                if w.chars().all(|c| c.is_ascii_uppercase()) && 1 < w.len() && w.len() <= 6 {
                    if let Some(ps) = spell_out(w) {
                        return Some(ps);
                    }
                }
                return None;
            }
            Tok::Number(_) | Tok::Other(_) => return None,
        }
    }
    None
}

/// Convert text to phonemes (Day-3 scope: words + punctuation; numbers are
/// reported gaps until Day 4).
pub fn g2p(text: &str) -> G2pResult {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let toks = tokenize(text);
    let mut parts: Vec<String> = Vec::new();
    let mut oov = Vec::new();
    let mut sent_start = true;

    for (idx, tok) in toks.iter().enumerate() {
        match tok {
            Tok::Punct(p) => {
                if parts.is_empty() {
                    parts.push(p.to_string());
                } else if let Some(last) = parts.last_mut() {
                    last.push(*p);
                }
            }
            Tok::Number(n) => {
                // Day 4: number expansion. Until then the digits are a
                // reported gap, not a silent one.
                oov.push(n.clone());
                parts.push(n.clone());
            }
            Tok::Other(c) => match c {
                // proto3 skips these outright (hyphen join, currency prefix
                // handled with numbers on Day 4, stray apostrophes)
                '\'' | '-' | '$' => {}
                // symbol words (proto3 SYMBOLS/ADD_SYMBOLS)
                '@' => parts.push("ˈæt".to_string()),
                '+' => parts.push("plˈʌs".to_string()),
                '/' => parts.push("slˈæʃ".to_string()),
                _ => parts.push(c.to_string()),
            },
            Tok::Word(w) => {
                let low = w.to_lowercase();
                // function-word context rules first (proto3 order)
                if w == "a" || (w == "A" && !sent_start) {
                    parts.push("ɐ".to_string());
                    continue;
                }
                if w == "A" && sent_start {
                    // 'A' at sentence start: article if a lowercase word
                    // follows, letter name otherwise
                    let article = matches!(toks.get(idx + 1), Some(Tok::Word(n2))
                        if n2.chars().next().map(|c| c.is_lowercase()).unwrap_or(false));
                    if article {
                        parts.push("ɐ".to_string());
                    } else {
                        parts.push("ˈA".to_string());
                    }
                    continue;
                }
                if low == "an" {
                    parts.push("ɐn".to_string());
                    continue;
                }
                if low == "the" {
                    // misaki: ði before vowel-SOUND-initial word, ðə otherwise
                    let nxt_ps = next_word_phonemes(idx, &toks);
                    let nxt_clean = nxt_ps
                        .as_deref()
                        .map(|s| s.trim_start_matches(['ˈ', 'ˌ']))
                        .unwrap_or("");
                    let vowel_start = nxt_clean
                        .chars()
                        .next()
                        .map(|c| "AEIOUaeiɔɑʊɪɛəɐᵻæ".contains(c))
                        .unwrap_or(false);
                    if vowel_start {
                        parts.push("ði".to_string());
                    } else {
                        parts.push("ðə".to_string());
                    }
                    continue;
                }
                if low == "this" {
                    parts.push("ðɪs".to_string());
                    continue;
                }
                if low == "to" {
                    // misaki: tə before any word (consonant, vowel, or number)
                    let mut nxt2: Option<&Tok> = toks.get(idx + 1);
                    if matches!(nxt2, Some(Tok::Other(c)) if *c == '-' || *c == '\u{2014}') {
                        nxt2 = toks.get(idx + 2);
                    }
                    let before_word = match nxt2 {
                        Some(Tok::Word(n2)) => n2.chars().next().is_some(),
                        Some(Tok::Number(_)) => true,
                        _ => false,
                    };
                    if before_word {
                        parts.push("tə".to_string());
                    } else {
                        parts.push("tu".to_string());
                    }
                    continue;
                }
                if low == "i" {
                    parts.push("ˌI".to_string());
                    continue;
                }
                if low == "in" {
                    parts.push("ɪn".to_string());
                    continue;
                }
                if low == "there" {
                    parts.push("ðˌɛɹ".to_string());
                    continue;
                }
                if low == "am" {
                    parts.push("ɐm".to_string());
                    continue;
                }
                if w == "That" && sent_start {
                    parts.push("ðˈæt".to_string());
                    sent_start = false;
                    continue;
                }
                // normal word: misaki, then CMU, then acronym spelling
                let mut ps = lexicon_lookup(w).map(|s| s.to_string());
                if ps.is_none() {
                    ps = cmu_lookup(&low);
                }
                if w.chars().all(|c| c.is_ascii_uppercase()) && 1 < w.len() && w.len() <= 6 {
                    if let Some(sp) = spell_out(w) {
                        ps = Some(sp.replace(' ', ""));
                    }
                }
                match ps {
                    None => {
                        // proto3 drops OOV words from the output and reports
                        // them; the stderr warning keeps this non-silent.
                        oov.push(w.clone());
                    }
                    Some(mut ps) => {
                        // paradee final pass: all ɾ -> T (lexicon words too)
                        ps = arpabet::flap_all(&ps);
                        // cap stress for capitalized non-special words
                        // (Python str.islower(): cased chars must be lower)
                        let first_upper =
                            w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
                        let rest: Vec<char> = w.chars().skip(1).collect();
                        let rest_lower = rest.iter().any(|c| c.is_alphabetic())
                            && rest
                                .iter()
                                .all(|c| !c.is_alphabetic() || c.is_lowercase());
                        if first_upper && rest_lower && !ps.contains('ˈ') && !ps.contains('ˌ') {
                            ps = arpabet::cap_stress(&ps);
                        }
                        parts.push(ps);
                    }
                }
                sent_start = false;
            }
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
        assert_eq!(r.phonemes, "həlˈO ðˌɛɹ.");
        assert!(r.oov.is_empty());
    }

    #[test]
    fn punctuation_glues() {
        let r = g2p("Wait, really?");
        assert!(r.phonemes.starts_with("wˈAt, ɹˈiᵊli?"), "{}", r.phonemes);
    }

    #[test]
    fn oov_is_reported_not_silent() {
        // proto3 semantics: OOV words are dropped from the phoneme string
        // but reported in `oov` (and warned on stderr by the CLI).
        let r = g2p("The zxqj sat.");
        assert_eq!(r.oov, vec!["zxqj".to_string()]);
        assert!(!r.phonemes.contains("zxqj"));
    }

    #[test]
    fn numbers_are_reported_until_day_4() {
        let r = g2p("I have 3 cats.");
        assert_eq!(r.oov, vec!["3".to_string()]);
    }

    #[test]
    fn cmu_fallback_word() {
        // "aardvark" is CMU-only: full ARPAbet -> misaki conversion path
        let r = g2p("The aardvark ran.");
        assert!(r.oov.is_empty(), "{:?}", r.oov);
        assert!(r.phonemes.contains("ˈɑɹdvˌɑɹk"), "{}", r.phonemes);
    }

    #[test]
    fn stem_suffix_functions() {
        // add_s: sibilant stems get IH0 Z, others get Z
        assert_eq!(add_s("K AE1 T"), Some("K AE1 T Z".to_string()));
        assert_eq!(add_s("B AH1 K S"), Some("B AH1 K S IH0 Z".to_string()));
        // add_ed: T/D stems get IH0 D, voiceless get T, else D
        assert_eq!(add_ed("W AO1 K"), Some("W AO1 K T".to_string()));
        assert_eq!(add_ed("W AO1 K D"), Some("W AO1 K D IH0 D".to_string()));
        assert_eq!(add_ed("S T AH1 P"), Some("S T AH1 P T".to_string()));
        // add_ing
        assert_eq!(add_ing("R AH1 N"), Some("R AH1 N IH0 NG".to_string()));
    }

    #[test]
    fn stem_rule_end_to_end() {
        // "running" is not in misaki; the doubled-consonant -ing rule finds
        // "run" in CMU and appends IH0 NG.
        let r = g2p("running");
        assert!(r.oov.is_empty(), "{:?}", r.oov);
        assert!(r.phonemes.contains("ɪŋ"), "{}", r.phonemes);
    }

    #[test]
    fn the_before_vowel_sound() {
        // misaki's "apple" is ˈæpᵊl (reduction vowel)
        assert_eq!(g2p("the apple").phonemes, "ði ˈæpᵊl");
        assert_eq!(g2p("the dog").phonemes, "ðə dˈɔɡ");
    }

    #[test]
    fn function_words() {
        assert_eq!(g2p("I am in it.").phonemes, "ˌI ɐm ɪn ɪt.");
        assert_eq!(g2p("to the store").phonemes, "tə ðə stˈɔɹ");
    }

    #[test]
    fn sentence_start_that() {
        assert!(g2p("That works.").phonemes.starts_with("ðˈæt"));
        // mid-sentence "that" takes the lexicon entry
        assert!(!g2p("I said that.").phonemes.starts_with("ðˈæt"));
    }

    #[test]
    fn capitalized_word_with_stress_unchanged() {
        // "Hello" already carries ˈ in misaki, so cap_stress must NOT fire
        // (proto3: only words with no stress marks get ˌ inserted; the
        // insertion itself is unit-tested in arpabet::tests).
        assert_eq!(g2p("Hello").phonemes, "həlˈO");
    }
}
