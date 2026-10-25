//! g2p CLI: read English text, print misaki phonemes.
//!
//! Usage:
//!   g2p "Hello there."          one text argument
//!   g2p                         reads stdin, one line per conversion
//!
//! OOV words are printed to stderr as a warning, never silently ignored.

use std::io::{BufRead, Write};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        convert_and_print(&args.join(" "));
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
        convert_and_print(&line);
    }
}

fn convert_and_print(text: &str) {
    let r = tinytts_g2p::g2p(text);
    println!("{}", r.phonemes);
    if !r.oov.is_empty() {
        let stderr = std::io::stderr();
        let mut w = stderr.lock();
        let _ = writeln!(w, "warning: OOV words: {}", r.oov.join(", "));
    }
}
