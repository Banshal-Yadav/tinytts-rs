//! Word pipeline: tokenization, misaki lookup, CMU ARPAbet fallback with
//! -s/-ed/-ing stem rules, function-word context rules, and the misaki-style
//! post passes (flap, flap_all, demote_later, cap_stress). Ported from the
//! validated Python prototype (g2p-spike/proto3.py); the differential test
//! (scripts/diff_g2p.py) enforces byte-exact agreement with that reference.
//!
//! Number tokens are reported as gaps, never silently dropped; number
//! expansion is a separate, not-yet-ported layer.

use crate::arpabet;
use crate::lexicon;
use crate::numbers;
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
    // Mirrors the reference TOKEN_RE:
    //   [A-Za-z]+(?:'[A-Za-z]+)? | \d+(?:[.,]\d+)*(?:st|nd|rd|th)? |
    //   [;:,.!?\u{2014}\u{2026}"\u{201c}\u{201d}()] | \S
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_alphabetic() {
            let mut w = String::new();
            while i < chars.len() && chars[i].is_ascii_alphabetic() {
                w.push(chars[i]);
                i += 1;
            }
            // optional internal apostrophe group: (?:'[A-Za-z]+)?
            if i < chars.len() && chars[i] == '\'' {
                let mut j = i + 1;
                let mut tail = String::new();
                while j < chars.len() && chars[j].is_ascii_alphabetic() {
                    tail.push(chars[j]);
                    j += 1;
                }
                if !tail.is_empty() {
                    w.push('\'');
                    w.push_str(&tail);
                    i = j;
                }
            }
            out.push(Tok::Word(w));
        } else if c.is_ascii_digit() {
            let mut n = String::new();
            while i < chars.len() && chars[i].is_ascii_digit() {
                n.push(chars[i]);
                i += 1;
            }
            // (?:[.,]\d+)*
            loop {
                if i + 1 < chars.len()
                    && (chars[i] == '.' || chars[i] == ',')
                    && chars[i + 1].is_ascii_digit()
                {
                    n.push(chars[i]);
                    i += 1;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        n.push(chars[i]);
                        i += 1;
                    }
                } else {
                    break;
                }
            }
            // (?:st|nd|rd|th)?
            for suf in ["st", "nd", "rd", "th"] {
                let sc: Vec<char> = suf.chars().collect();
                if i + 2 <= chars.len() && chars[i..i + 2] == sc[..] {
                    n.push_str(suf);
                    i += 2;
                    break;
                }
            }
            out.push(Tok::Number(n));
        } else if is_punct(c) {
            out.push(Tok::Punct(c));
            i += 1;
        } else if c.is_whitespace() {
            i += 1;
        } else {
            out.push(Tok::Other(c));
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CMU fallback with stem rules (cmu_lookup / add_s / add_ed / add_ing)
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

/// CMUdict lookup with the stem rules. `word` must be lowercase.
fn cmu_lookup(word: &str) -> Option<String> {
    let mut phones = lexicon::lookup_cmu(word).map(|s| s.to_string());
    if phones.is_none() {
        // -s: cats -> cat, boxes -> box (via -es), cities -> city
        if word.len() > 2 && word.ends_with('s') && !word.ends_with("ss") {
            let mut stems: Vec<String> = vec![word[..word.len() - 1].to_string()];
            if let Some(s) = word.strip_suffix("es") {
                stems.push(s.to_string());
            }
            if let Some(s) = word.strip_suffix("ies") {
                stems.push(format!("{}y", s));
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
            // reference regex: ([bcdgklmnprstvxz])\1ing$ or cking$
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
/// byte compare; the lowercase fallback mirrors the reference lookup).
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

/// misaki Lexicon._s (US English):
/// if stem ends in voiceless consonant: + 's'
/// if stem ends in sibilant (s, z, ʃ, ʒ, ʧ, ʤ): + 'ᵻz'
/// else: + 'z'
fn misaki_s(stem_ps: &str) -> String {
    let mut out = stem_ps.to_string();
    let last = match stem_ps.chars().last() {
        Some(c) => c,
        None => return out,
    };
    if matches!(last, 'p' | 't' | 'k' | 'f' | 'θ') {
        out.push('s');
    } else if matches!(last, 's' | 'z' | 'ʃ' | 'ʒ' | 'ʧ' | 'ʤ') {
        out.push_str("ᵻz");
    } else {
        out.push('z');
    }
    out
}

/// Try splitting an OOV compound word into two known lexicon words.
/// E.g. stonemasons -> stone + masons, treehouse -> tree + house.
fn split_compound(word: &str) -> Option<String> {
    let w = word.to_lowercase();
    if w.len() < 6 || !w.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    for i in 3..w.len() - 2 {
        let w1 = &w[..i];
        let w2 = &w[i..];
        let ps1 = match lexicon_lookup(w1).map(|s| s.to_string()).or_else(|| cmu_lookup(w1)) {
            Some(p) => p,
            None => continue,
        };
        let mut ps2 = lexicon_lookup(w2).map(|s| s.to_string()).or_else(|| cmu_lookup(w2));
        if ps2.is_none() && w2.ends_with('s') && !w2.ends_with("ss") {
            let stem2 = &w2[..w2.len() - 1];
            if let Some(p2) = lexicon_lookup(stem2).map(|s| s.to_string()).or_else(|| cmu_lookup(stem2)) {
                ps2 = Some(misaki_s(&p2));
            }
        }
        if let Some(ps2) = ps2 {
            let joined = format!("{}{}", ps1, ps2);
            return Some(arpabet::demote_later(&arpabet::flap_all(&joined)));
        }
    }
    None
}

/// Simple rule-based English letter-to-sound fallback for OOV words.
/// Never drops a word from output.
fn letter_to_sound(word: &str) -> Option<String> {
    const CONSONANTS: &[(&str, &str)] = &[
        ("th", "θ"), ("sh", "ʃ"), ("ch", "ʧ"), ("ph", "f"), ("qu", "kw"), ("wh", "w"),
        ("b", "b"), ("c", "k"), ("d", "d"), ("f", "f"), ("g", "ɡ"), ("h", "h"),
        ("j", "ʤ"), ("k", "k"), ("l", "l"), ("m", "m"), ("n", "n"), ("p", "p"),
        ("r", "ɹ"), ("s", "s"), ("t", "t"), ("v", "v"), ("w", "w"), ("x", "ks"),
        ("z", "z"),
    ];
    const VOWELS_MAP: &[(&str, &str)] = &[
        ("a", "æ"), ("e", "ɛ"), ("i", "ɪ"), ("o", "ɑ"), ("u", "ʌ"), ("y", "i"),
    ];

    let w = word.to_lowercase();
    let mut out = String::new();
    let mut i = 0;
    let mut first_vowel = true;
    while i < w.len() {
        let rest = &w[i..];
        let mut matched = false;
        for &(pat, ph) in CONSONANTS {
            if rest.starts_with(pat) {
                out.push_str(ph);
                i += pat.len();
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        for &(pat, ph) in VOWELS_MAP {
            if rest.starts_with(pat) {
                if first_vowel {
                    out.push('ˈ');
                    first_vowel = false;
                }
                out.push_str(ph);
                i += pat.len();
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        let c = rest.chars().next()?;
        if c.is_ascii_alphabetic() {
            if let Some(sp) = spell_out(&c.to_string()) {
                out.push_str(&sp);
            }
        }
        i += c.len_utf8();
    }
    if out.is_empty() {
        None
    } else {
        Some(arpabet::flap_all(&out))
    }
}

/// Lexicon first, then CMU fallback. Used by the number-expansion path.
fn word_phonemes_any(word: &str) -> Option<String> {
    if let Some(ps) = lexicon_lookup(word) {
        return Some(ps.to_string());
    }
    cmu_lookup(&word.to_lowercase())
}

/// Phonemes for one word of a number expansion. misaki looks up "point" with
/// stress -2 (all stress marks stripped).
fn word_phonemes_num(word: &str) -> Option<String> {
    let mut ps = word_phonemes_any(word)?;
    if word == "point" {
        ps = ps.replace(['ˈ', 'ˌ'], "");
    }
    Some(ps)
}

// ---------------------------------------------------------------------------
// Function-word context rules (word-level branch)
// ---------------------------------------------------------------------------

/// Letter-by-letter spelling using misaki's gold letter names (the
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
    for tok in toks.iter().take((idx + 3).min(toks.len())).skip(idx + 1) {
        match tok {
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
            Tok::Number(n) => {
                if let Some(words) = numbers::number_words(n) {
                    if let Some(first) = words.first() {
                        return lexicon_lookup(first)
                            .map(|s| s.to_string())
                            .or_else(|| cmu_lookup(&first.to_lowercase()));
                    }
                }
                return None;
            }
            Tok::Other(_) => return None,
        }
    }
    None
}

/// Convert text to phonemes. Words and punctuation are fully handled;
/// number tokens are reported as gaps until number expansion is ported.
pub fn g2p(text: &str) -> G2pResult {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let toks = tokenize(text);
    let mut parts: Vec<String> = Vec::new();
    let mut oov = Vec::new();
    let mut sent_start = true;

    for (idx, tok) in toks.iter().enumerate() {
        let nxt = toks.get(idx + 1);
        match tok {
            Tok::Punct(p) => {
                if parts.is_empty() {
                    parts.push(p.to_string());
                } else if let Some(last) = parts.last_mut() {
                    last.push(*p);
                }
            }
            Tok::Number(n) => {
                // ordinal suffix: 26th -> twenty sixth
                if let Some(ow) = numbers::ordinal_words(n) {
                    let mut ok = true;
                    for w in ow.split(' ') {
                        match word_phonemes_any(w) {
                            Some(ps) => parts.push(arpabet::flap_all(&ps)),
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        oov.push(n.clone());
                    }
                    continue;
                }
                // currency prefix: $18 -> eighteen dollars, $4.7 -> four dollars
                // and seven cents, $0.99 -> ninety nine cents
                if matches!(toks.get(idx.wrapping_sub(1)), Some(Tok::Other('$'))) {
                    let words = numbers::currency_words(n);
                    let mut ok = true;
                    for w in &words {
                        match word_phonemes_num(w) {
                            Some(ps) => parts.push(arpabet::flap_all(&ps)),
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        oov.push(n.clone());
                    }
                    continue;
                }
                let words = match numbers::number_words(n) {
                    Some(w) => w,
                    None => {
                        oov.push(n.clone());
                        continue;
                    }
                };
                let mut ok = true;
                for w in &words {
                    match word_phonemes_num(w) {
                        Some(ps) => parts.push(arpabet::flap_all(&ps)),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    oov.push(n.clone());
                }
            }
            Tok::Other(c) => {
                // negative number: '-' before a digit token -> minus
                if *c == '-'
                    && matches!(nxt, Some(Tok::Number(n2)) if n2.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false))
                {
                    parts.push("mˈInəs".to_string());
                    continue;
                }
                match c {
                    // skipped outright (hyphen join, currency prefix handled
                    // with numbers, stray apostrophes)
                    '\'' | '-' | '\u{2014}' | '$' => {}
                    // date separator: '/' between two digit tokens is dropped
                    // (12/3/1905 -> twelve three nineteen oh five)
                    '/' if matches!(toks.get(idx.wrapping_sub(1)), Some(Tok::Number(p))
                            if p.bytes().all(|b| b.is_ascii_digit()))
                        && matches!(nxt, Some(Tok::Number(n2))
                            if n2.bytes().all(|b| b.is_ascii_digit())) => {}
                    // symbol words (reference SYMBOLS/ADD_SYMBOLS)
                    '@' => parts.push("ˈæt".to_string()),
                    '%' => parts.push("pəɹsˈɛnt".to_string()),
                    '+' => parts.push("plˈʌs".to_string()),
                    '/' => parts.push("slˈæʃ".to_string()),
                    // any other non-space char is treated as a word: lexicon,
                    // then CMU, then OOV (matches the reference fallthrough)
                    _ => {
                        let s = c.to_string();
                        match word_phonemes_any(&s) {
                            Some(ps) => parts.push(arpabet::flap_all(&ps)),
                            None => oov.push(s),
                        }
                    }
                }
            }
            Tok::Word(w) => {
                let low = w.to_lowercase();
                // function-word context rules first (reference order)
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
                    // misaki: {None: 'tu', False: 'tə', True: 'tʊ'}[ctx.future_vowel]
                    let nxt_ps = next_word_phonemes(idx, &toks);
                    if let Some(np) = nxt_ps {
                        let clean = np.trim_start_matches(['ˈ', 'ˌ']);
                        let vowel_start = clean
                            .chars()
                            .next()
                            .map(|c| "AEIOUaeiɔɑʊɪɛəɐᵻæ".contains(c))
                            .unwrap_or(false);
                        if vowel_start {
                            parts.push("tʊ".to_string());
                        } else {
                            parts.push("tə".to_string());
                        }
                    } else {
                        parts.push("tu".to_string());
                    }
                    continue;
                }
                if low == "used" {
                    let is_to = matches!(toks.get(idx + 1), Some(Tok::Word(w2)) if w2.eq_ignore_ascii_case("to"));
                    if is_to {
                        parts.push("jˈust".to_string());
                        continue;
                    }
                }
                if low == "i" {
                    parts.push("ˌI".to_string());
                    continue;
                }
                if low == "in" {
                    // misaki: ˈɪn when stressed (sentence-final / before punct), ɪn otherwise
                    let nxt_ps = next_word_phonemes(idx, &toks);
                    if nxt_ps.is_none() {
                        parts.push("ˈɪn".to_string());
                    } else {
                        parts.push("ɪn".to_string());
                    }
                    continue;
                }
                if low == "there" {
                    // misaki: ðˌɛɹ DEFAULT, ðˈɛɹ with None tag (sentence-final or copula)
                    let nxt_ps = next_word_phonemes(idx, &toks);
                    if nxt_ps.is_none() {
                        parts.push("ðˈɛɹ".to_string());
                    } else {
                        parts.push("ðˌɛɹ".to_string());
                    }
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
                // normal word: misaki, then possessive via misaki stem, then CMU
                let mut ps = lexicon_lookup(w).map(|s| s.to_string());
                // possessive 's against misaki lexicon first: boss's -> boss + ᵻz
                if ps.is_none() && w.ends_with("'s") && w.len() > 2 {
                    let stem = &w[..w.len() - 2];
                    if let Some(sp) = lexicon_lookup(stem) {
                        ps = Some(misaki_s(sp));
                    }
                }
                // plural possessive: teachers' -> teachers
                if ps.is_none() && w.ends_with("s'") && w.len() > 2 {
                    let stem = &w[..w.len() - 1];
                    if let Some(sp) = lexicon_lookup(stem) {
                        ps = Some(sp.to_string());
                    }
                }
                if ps.is_none() {
                    ps = cmu_lookup(&low);
                }
                // cmu-based possessive
                if ps.is_none() && w.ends_with("'s") && w.len() > 2 {
                    let stem = &w[..w.len() - 2];
                    if let Some(sp) = cmu_lookup(&stem.to_lowercase()) {
                        ps = Some(misaki_s(&sp));
                    }
                }
                if ps.is_none() && w.ends_with("s'") && w.len() > 2 {
                    let stem = &w[..w.len() - 1];
                    if let Some(sp) = cmu_lookup(&stem.to_lowercase()) {
                        ps = Some(sp);
                    }
                }
                // compound splitting: stonemasons -> stone + masons
                if ps.is_none() && w.len() >= 6 && w.chars().all(|c| c.is_ascii_alphabetic()) {
                    ps = split_compound(w);
                }
                if w.chars().all(|c| c.is_ascii_uppercase()) && 1 < w.len() && w.len() <= 6 {
                    if let Some(sp) = spell_out(w) {
                        ps = Some(sp.replace(' ', ""));
                    }
                }
                // OOV fallback: rule-based letter-to-sound (never drop a word!)
                if ps.is_none() {
                    oov.push(w.clone());
                    ps = letter_to_sound(w);
                }
                let Some(mut ps) = ps else {
                    continue;
                };
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
        assert_eq!(r.phonemes, "həlˈO ðˈɛɹ.");
        assert!(r.oov.is_empty());
    }

    #[test]
    fn possessive_and_compound_rules() {
        // bakery's -> bakery + 's -> bˈAkəɹiz
        let r = g2p("the bakery's bread");
        assert!(r.oov.is_empty(), "{:?}", r.oov);
        assert!(r.phonemes.contains("bˈAkəɹiz"), "{}", r.phonemes);

        // boss's -> boss + sibilant -> bˈɔsᵻz
        let r = g2p("the boss's desk");
        assert!(r.oov.is_empty(), "{:?}", r.oov);
        assert!(r.phonemes.contains("bˈɔsᵻz"), "{}", r.phonemes);

        // compound splitting: stonemasons -> stone + masons
        let r = g2p("local stonemasons");
        assert!(r.oov.is_empty(), "{:?}", r.oov);
        assert!(r.phonemes.contains("stˈOnmˌAsᵊnz"), "{}", r.phonemes);

        // OOV letter-to-sound: never dropped!
        let r = g2p("zxqj");
        assert_eq!(r.oov, vec!["zxqj".to_string()]);
        assert!(!r.phonemes.is_empty(), "OOV word was dropped!");
    }

    #[test]
    fn context_rules_to_and_used() {
        // to before vowel sound -> tʊ
        assert_eq!(g2p("to eat").phonemes, "tʊ ˈit");
        // to before consonant sound -> tə
        assert_eq!(g2p("to work").phonemes, "tə wˈɜɹk");
        // used to -> jˈust tə
        assert_eq!(g2p("used to go").phonemes, "jˈust tə ɡˌO");
    }

    #[test]
    fn punctuation_glues() {
        let r = g2p("Wait, really?");
        assert!(r.phonemes.starts_with("wˈAt, ɹˈiᵊli?"), "{}", r.phonemes);
    }

    #[test]
    fn oov_is_reported_not_silent() {
        // reference semantics: OOV words are dropped from the phoneme string
        // but reported in `oov` (and warned on stderr by the CLI).
        let r = g2p("The zxqj sat.");
        assert_eq!(r.oov, vec!["zxqj".to_string()]);
        assert!(!r.phonemes.contains("zxqj"));
    }

    #[test]
    fn numbers_expand() {
        let r = g2p("I have 3 cats.");
        assert!(r.oov.is_empty(), "{:?}", r.oov);
        assert!(r.phonemes.contains("θɹˈi"), "{}", r.phonemes);
    }

    #[test]
    fn known_findings() {
        // 2019 -> "twenty nineteen" (not "twenty nine")
        assert!(g2p("In 2019 we left.").phonemes.contains("twˈɛnti nˌIntˈin"));
        // 40th -> "fortieth" (not "fortyth")
        assert!(g2p("He came 40th.").phonemes.contains("fˈɔɹTiəθ"));
        // 1,284.67 -> thousands + decimal (point is unstressed like misaki)
        let r = g2p("It cost 1,284.67 dollars.");
        assert!(r.oov.is_empty(), "{:?}", r.oov);
        assert!(r.phonemes.contains("wˈʌn θˈWzᵊnd"), "{}", r.phonemes);
        assert!(r.phonemes.contains("pYnt"), "{}", r.phonemes);
        // percent sign maps to "percent" (misaki SYMBOLS)
        let r = g2p("rose 8.3%");
        assert!(r.oov.is_empty(), "{:?}", r.oov);
        assert!(r.phonemes.contains("pəɹsˈɛnt"), "{}", r.phonemes);
    }

    #[test]
    fn date_and_time() {
        // 12/3/1905 -> twelve three nineteen oh five (slashes dropped)
        let r = g2p("on 12/3/1905");
        assert!(!r.phonemes.contains("slˈæʃ"), "{}", r.phonemes);
        assert!(r.phonemes.contains("twˈɛlv θɹˈi"), "{}", r.phonemes);
        // 9:30 -> nine thirty (colon glued)
        assert!(g2p("at 9:30").phonemes.contains("nˈIn: θˈɜɹTi"));
    }

    #[test]
    fn negative_and_currency() {
        assert!(g2p("was -7 degrees").phonemes.contains("mˈInəs sˈɛvən"));
        assert!(g2p("was -1,000 degrees").phonemes.contains("mˈInəs wˈʌn θˈWzᵊnd"));
        assert!(g2p("costs $18").phonemes.contains("ˌAtˈin dˈɑləɹz"));
        assert!(g2p("costs $4.7").phonemes.contains("fˈɔɹ dˈɑləɹz ænd sˈɛvən sˈɛnts"));
        assert!(g2p("costs $0.99").phonemes.contains("nˈIndi nˈIn sˈɛnts"));
    }

    #[test]
    fn ordinals_and_years() {
        assert!(g2p("finished 12th").phonemes.contains("twˈɛlfθ"));
        assert!(g2p("in 1000").phonemes.contains("wˈʌn θˈWzᵊnd"));
        assert!(g2p("in 1001").phonemes.contains("wˈʌn θˈWzᵊnd wˈʌn"));
        assert!(g2p("in 2000").phonemes.contains("tˈu θˈWzᵊnd"));
        assert!(g2p("in 2005").phonemes.contains("tˈu θˈWzᵊnd fˈIv"));
        assert!(g2p("in 2100").phonemes.contains("twˈɛnti wˈʌn hˈʌndɹəd"));
        assert!(g2p("42.50").phonemes.contains("fˈɔɹTi tˈu pYnt fˈIv"));
        assert_eq!(g2p("4.0").phonemes, "fˈɔɹ");
        assert_eq!(g2p("100.00").phonemes, "wˈʌn hˈʌndɹəd");
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
        // (only words with no stress marks get a secondary mark inserted; the
        // insertion itself is unit-tested in arpabet::tests).
        assert_eq!(g2p("Hello").phonemes, "həlˈO");
    }
}
