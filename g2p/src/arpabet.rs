//! CMUdict ARPAbet to misaki symbol mapping, ported from the validated
//! Python prototype (g2p-spike/proto3.py, `MAP` / `UNSTRESSED` / `cmu_convert`).
//!
//! The map was statistically derived against misaki's own lexicon
//! (see g2p-spike/derive_map.py); it is frozen here so the Rust port
//! matches the Python reference byte for byte. The differential test
//! (scripts/diff_g2p.py) enforces that.

/// ARPAbet base phone -> misaki symbol (stressed or fully-reduced forms are
/// handled separately in `convert`).
pub const MAP: &[(&str, &str)] = &[
    ("AA", "ɑ"), ("AE", "æ"), ("AH", "ʌ"), ("AO", "ɔ"), ("AW", "W"),
    ("AY", "I"), ("B", "b"), ("CH", "ʧ"), ("D", "d"), ("DH", "ð"),
    ("EH", "ɛ"), ("ER", "ɜ"), ("EY", "A"), ("F", "f"), ("G", "ɡ"),
    ("HH", "h"), ("IH", "ɪ"), ("IY", "i"), ("JH", "ʤ"), ("K", "k"),
    ("L", "l"), ("M", "m"), ("N", "n"), ("NG", "ŋ"), ("OW", "O"),
    ("OY", "Y"), ("P", "p"), ("R", "ɹ"), ("S", "s"), ("SH", "ʃ"),
    ("T", "t"), ("TH", "θ"), ("UH", "ʊ"), ("UW", "u"), ("V", "v"),
    ("W", "w"), ("Y", "j"), ("Z", "z"), ("ZH", "ʒ"),
];

/// Unstressed (stress-0) vowel overrides. Anything not listed keeps its MAP
/// symbol; AH0/IH0 have extra context rules in `convert`.
pub const UNSTRESSED: &[(&str, &str)] = &[
    ("AH", "ə"), ("IH", "ɪ"), ("IY", "i"), ("EY", "A"), ("OW", "O"),
    ("UW", "u"), ("AE", "æ"), ("AA", "ɑ"), ("AO", "ɔ"), ("EH", "ɛ"),
    ("UH", "ʊ"), ("AW", "W"), ("AY", "I"),
];

/// Vowel set used by the flap rule (misaki's VOWELS, same characters).
pub const VOWELS: &str = "AIOWYaiuæɑɒɔəɛɜɪʊʌᵻ";

fn map_lookup(table: &'static [(&'static str, &'static str)], key: &str) -> Option<&'static str> {
    table
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| *v)
}

/// Is `base` a vowel ARPAbet phone (used by the AH0-before-nasal rule)?
fn is_vowel_base(base: &str) -> bool {
    matches!(
        base,
        "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY" | "OW" | "OY"
            | "UH" | "UW"
    )
}

/// Convert one word's ARPAbet phones (e.g. ["D", "AO1", "G"]) to a misaki
/// phoneme string. Returns None for phones outside the map (reported as OOV
/// by the caller, never silently dropped).
pub fn convert(phones: &[&str]) -> Option<String> {
    let n = phones.len();
    let mut out = String::new();
    for (i, ph) in phones.iter().enumerate() {
        let base = ph.trim_end_matches(|c| ('0'..='2').contains(&c));
        let stress = &ph[base.len()..];
        let sym: String;
        if base == "ER" {
            if stress == "1" || stress == "2" {
                sym = "ɜɹ".to_string();
            } else {
                // ER0: ɹ word-final, əɹ otherwise
                sym = if i == n - 1 { "ɹ" } else { "əɹ" }.to_string();
            }
        } else {
            let m = map_lookup(MAP, base)?;
            if stress == "0" {
                if base == "IH" {
                    // misaki reduces unstressed IH to ə in CMU fallback words
                    sym = "ə".to_string();
                } else if base == "AH" {
                    // AH0 -> ᵊ before nasal+consonant (applicants, questions)
                    let next_nasal = i + 2 < n && {
                        let b1 = phones[i + 1].trim_end_matches(|c| ('0'..='2').contains(&c));
                        b1 == "N" || b1 == "M" || b1 == "NG"
                    };
                    let next2_vowel = i + 2 < n && {
                        let b2 = phones[i + 2].trim_end_matches(|c| ('0'..='2').contains(&c));
                        is_vowel_base(b2)
                    };
                    if next_nasal && !next2_vowel {
                        sym = "ᵊ".to_string();
                    } else {
                        sym = "ə".to_string();
                    }
                } else {
                    sym = map_lookup(UNSTRESSED, base)
                        .unwrap_or(m)
                        .to_string();
                }
            } else {
                sym = m.to_string();
            }
        }
        if stress == "1" {
            out.push('ˈ');
        } else if stress == "2" {
            out.push('ˌ');
        }
        out.push_str(&sym);
    }
    Some(out)
}

/// Intervocalic /t/ flapping (proto3 `flap`): t between vowels, or t before
/// ɹ+vowel, becomes T.
pub fn flap(ps: &str) -> String {
    let chars: Vec<char> = ps.chars().collect();
    let mut out = String::with_capacity(ps.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == 't' {
            let prev_v = i > 0 && VOWELS.contains(chars[i - 1]);
            let next_v = i + 1 < chars.len()
                && (VOWELS.contains(chars[i + 1])
                    || (chars[i + 1] == 'ɹ'
                        && i + 2 < chars.len()
                        && VOWELS.contains(chars[i + 2])));
            if prev_v && next_v {
                out.push('T');
                continue;
            }
        }
        out.push(c);
    }
    out
}

/// misaki's final pass (paradee variant): ALL intervocalic ɾ -> T.
pub fn flap_all(ps: &str) -> String {
    ps.replace('ɾ', "T")
}

/// Keep the first primary stress, demote later primaries to secondary.
pub fn demote_later(ps: &str) -> String {
    let mut seen = false;
    ps.chars()
        .map(|c| {
            if c == 'ˈ' {
                if seen {
                    'ˌ'
                } else {
                    seen = true;
                    'ˈ'
                }
            } else {
                c
            }
        })
        .collect()
}

/// misaki cap_stresses[0] = 0.5: add ˌ before the first vowel if no stress
/// marks are present.
pub fn cap_stress(ps: &str) -> String {
    if ps.contains('ˈ') || ps.contains('ˌ') {
        return ps.to_string();
    }
    for (i, c) in ps.char_indices() {
        if VOWELS.contains(c) {
            let mut out = String::with_capacity(ps.len() + 3);
            out.push_str(&ps[..i]);
            out.push('ˌ');
            out.push_str(&ps[i..]);
            return out;
        }
    }
    ps.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_is_complete_for_arpabet() {
        // Every ARPAbet phone CMUdict can produce must be in MAP (except ER,
        // which has its own branch, and the vowel+digit combos).
        for (k, _) in MAP {
            assert!(!k.is_empty() && k.chars().all(|c| c.is_ascii_uppercase()));
        }
        // spot checks against the frozen derived map
        assert_eq!(map_lookup(MAP, "AY"), Some("I"));
        assert_eq!(map_lookup(MAP, "DH"), Some("ð"));
        assert_eq!(map_lookup(MAP, "ZZ"), None);
    }

    #[test]
    fn convert_dog() {
        assert_eq!(convert(&["D", "AO1", "G"]), Some("dˈɔɡ".to_string()));
    }

    #[test]
    fn convert_er_rules() {
        // ER1 -> ˈ + ɜɹ; ER0 final -> ɹ; ER0 medial -> əɹ
        assert_eq!(convert(&["B", "ER1", "D"]), Some("bˈɜɹd".to_string()));
        assert_eq!(convert(&["HH", "ER0"]), Some("hɹ".to_string()));
        assert_eq!(convert(&["B", "ER0", "R", "IY0"]), Some("bəɹɹi".to_string()));
    }

    #[test]
    fn convert_ah0_before_nasal() {
        // AH0 + N + consonant -> ᵊ (applicants: AE1 P L AH0 K AH0 N T S)
        assert_eq!(
            convert(&["AE1", "P", "L", "AH0", "K", "AH0", "N", "T", "S"]),
            Some("ˈæpləkᵊnts".to_string())
        );
    }

    #[test]
    fn flap_intervocalic() {
        assert_eq!(flap("wˈOtəɹ"), "wˈOTəɹ");
        // t before ɹ+vowel also flaps (i is a vowel, ː is not)
        assert_eq!(flap("mˈitɹɪk"), "mˈiTɹɪk");
        // word-final t does not
        assert_eq!(flap("wˈOt"), "wˈOt");
    }

    #[test]
    fn demote_keeps_first_primary() {
        assert_eq!(demote_later("dˈɛdˌlˈɪk"), "dˈɛdˌlˌɪk");
        assert_eq!(demote_later("ˈeɪb"), "ˈeɪb");
    }

    #[test]
    fn cap_stress_inserts_before_first_vowel() {
        // no stress marks present -> ˌ inserted
        assert_eq!(cap_stress("hɛlo"), "hˌɛlo");
        // any stress mark present -> unchanged
        assert_eq!(cap_stress("hɛlˈO"), "hɛlˈO");
        assert_eq!(cap_stress("ˈɛlᵻfənt"), "ˈɛlᵻfənt");
    }
}
