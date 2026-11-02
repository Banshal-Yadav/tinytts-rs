//! g2p CLI: read English text, print misaki phonemes or model token IDs.
//!
//! Usage:
//!   g2p "Hello there."          one text argument -> prints phonemes
//!   g2p --ids "Hello there."    prints model token IDs for tinytts: 0 50 83 ... 0
//!   g2p                         reads stdin, one line per conversion
//!   g2p --ids                   reads stdin, prints IDs for each line
//!
//! OOV words are printed to stderr as a warning, never silently ignored.

use std::io::{BufRead, Write};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut ids_mode = false;
    let mut texts = Vec::new();
    for a in args {
        if a == "--ids" || a == "--tokens" {
            ids_mode = true;
        } else {
            texts.push(a);
        }
    }

    if !texts.is_empty() {
        convert_and_print(&texts.join(" "), ids_mode);
        return;
    }
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("g2p: stdin read error: {}", e);
                std::process::exit(1);
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        convert_and_print(&line, ids_mode);
    }
}

fn convert_and_print(text: &str, ids_mode: bool) {
    let r = tinytts_g2p::g2p(text);
    if ids_mode {
        let ids = tinytts_g2p::vocab::phonemes_to_ids(&r.phonemes);
        let s = ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        println!("{}", s);
    } else {
        println!("{}", r.phonemes);
    }
    if !r.oov.is_empty() {
        let stderr = std::io::stderr();
        let mut w = stderr.lock();
        let _ = writeln!(w, "warning: OOV words: {}", r.oov.join(", "));
    }
}
