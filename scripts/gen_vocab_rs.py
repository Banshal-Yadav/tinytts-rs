import sys

with open('vocab.txt', encoding='utf-8') as f:
    entries = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        tid, ch = line.split('\t')
        ch = ch.strip("'")
        entries.append((ch, int(tid)))

entries.sort(key=lambda x: ord(x[0]))

code_lines = [
    "//! Vocab mapping from IPA characters to model token IDs.",
    "//!",
    "//! Generated from repo root `vocab.txt`.",
    "",
    "pub const VOCAB_TABLE: &[(char, i64)] = &[",
]

for c, tid in entries:
    if c == "'":
        c_repr = r"'\''"
    elif c == "\\":
        c_repr = r"'\\'"
    elif ord(c) < 128:
        c_repr = f"'{c}'"
    else:
        c_repr = f"'{c}'"
    code_lines.append(f"    ({c_repr}, {tid}),")

code_lines.extend([
    "];",
    "",
    "/// Look up token ID for a single character.",
    "pub fn char_to_id(c: char) -> Option<i64> {",
    "    VOCAB_TABLE",
    "        .binary_search_by_key(&c, |&(ch, _)| ch)",
    "        .ok()",
    "        .map(|idx| VOCAB_TABLE[idx].1)",
    "}",
    "",
    "/// Maximum phonemes model accepts before BOS/EOS (512 total - 2).",
    "pub const MAX_PHONEMES: usize = 510;",
    "",
    "/// Map a phoneme string to sequence of token IDs, bounded by BOS (0) and EOS (0).",
    "pub fn phonemes_to_ids(phonemes: &str) -> Vec<i64> {",
    "    let mut ids = Vec::with_capacity(phonemes.len() + 2);",
    "    ids.push(0);",
    "    for c in phonemes.chars() {",
    "        if let Some(id) = char_to_id(c) {",
    "            if ids.len() <= MAX_PHONEMES {",
    "                ids.push(id);",
    "            }",
    "        }",
    "    }",
    "    ids.push(0);",
    "    ids",
    "}",
    "",
    "#[cfg(test)]",
    "mod tests {",
    "    use super::*;",
    "",
    "    #[test]",
    "    fn test_vocab_lookup() {",
    "        assert_eq!(char_to_id(';'), Some(1));",
    "        assert_eq!(char_to_id(' '), Some(16));",
    "        assert_eq!(char_to_id('“'), Some(14));",
    "        assert_eq!(char_to_id('”'), Some(15));",
    "        assert_eq!(char_to_id('ᵻ'), Some(177));",
    "        assert_eq!(char_to_id('@'), None);",
    "    }",
    "",
    "    #[test]",
    "    fn test_phonemes_to_ids() {",
    "        let ids = phonemes_to_ids(\"həlˈO\");",
    "        assert_eq!(ids.first(), Some(&0));",
    "        assert_eq!(ids.last(), Some(&0));",
    "    }",
    "}",
])

with open('g2p/src/vocab.rs', 'w', encoding='utf-8') as out:
    out.write('\n'.join(code_lines) + '\n')

print(f"Generated g2p/src/vocab.rs with {len(entries)} tokens.")
