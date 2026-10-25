use std::time::Instant;
use tinytts::{exec, graph, onnx, tensor};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse flags and positionals in one pass. A flag that takes a value must
    // consume the next argument, otherwise that value would be mistaken for a
    // positional (the model path or a token id).
    let mut profile = false;
    let mut no_chunk = false;
    let mut stream = false;
    let mut shapes = false;
    let mut chunk_max: Option<usize> = None;
    let mut crossfade_ms: f64 = 10.0;
    let mut rng_seed: Option<u64> = None;
    let mut pos: Vec<&str> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        let take = |i: &mut usize| -> Option<&str> {
            *i += 1;
            args.get(*i).map(|s| s.as_str())
        };
        match a {
            "--profile" => profile = true,
            "--no-chunk" => no_chunk = true,
            "--stream" => stream = true,
            "--shapes" => shapes = true,
            "--chunk" => chunk_max = take(&mut i).and_then(|s| s.parse().ok()),
            "--crossfade" => {
                if let Some(v) = take(&mut i).and_then(|s| s.parse().ok()) {
                    crossfade_ms = v;
                }
            }
            // `--rng-seed N` fixes the xorshift seed used by `RandomNormalLike`,
            // so two runs on the same input produce byte-identical audio. Without
            // it the graph draws its own noise and the noisy parts of the
            // waveform differ run to run.
            "--rng-seed" => rng_seed = take(&mut i).and_then(|s| s.parse().ok()),
            _ => pos.push(a),
        }
        i += 1;
    }
    let path = *pos.first().expect(
        "usage: tinytts <model.onnx> [--profile] [--chunk N] [--crossfade MS] [--rng-seed N] [ids...]",
    );

    // Chunking is **opt-in and automatic only for long inputs**. `--chunk N`
    // forces it with a chunk size of N; `--no-chunk` disables it; the default is
    // to chunk only when the input exceeds `AUTO_CHUNK_TOKENS`.
    //
    // Chunking caps peak RSS at the single-chunk level and gets the first audio
    // out after one chunk, but it costs total synth time (a fixed per-chunk
    // encoder/LSTM cost) and it changes the pacing: each chunk is a separate
    // utterance, so the duration predictor gives it its own rhythm and the total
    // audio grows. Short inputs must therefore stay unchunked.
    const AUTO_CHUNK_TOKENS: usize = 100;
    const DEFAULT_CHUNK: usize = 100;

    let t0 = Instant::now();
    let g = {
        let bytes = std::fs::read(path).expect("read model");
        onnx::parse(&bytes)
    };
    let m = graph::Model::new(g);
    let t_load = t0.elapsed();
    eprintln!(
        "parse+topo: {:?}  nodes={} inits={}",
        t_load,
        m.g.nodes.len(),
        m.g.inits.len()
    );
    eprintln!("rss_after_load_mb: {:.1}", rss_mb());

    let ids: Vec<i64> = if pos.len() > 1 {
        pos[1..].iter().map(|s| s.parse().unwrap()).collect()
    } else {
        vec![0, 16, 43, 44, 45, 16, 4, 0]
    };
    let mut e = exec::Exec::new();
    e.profile = profile;
    if let Some(s) = rng_seed {
        e.rng = tinytts::quant::Rng::new(s);
    }

    // `--shapes`: dump every node's input/output shapes and exit. Used by
    // scripts/bench/gflops.py to compute exact MAC counts.
    if shapes {
        let n = ids.len();
        e.run_dump_shapes(
            &m,
            vec![
                ("input_ids".into(), tensor::T::from_i64(vec![1, n], ids)),
                ("speed".into(), tensor::T::from_f32(vec![1], vec![1.0])),
            ],
        );
        return;
    }

    // Decide whether to chunk: explicit flag wins, otherwise only for long input.
    // The model itself caps at 512 tokens, so an unchunked run longer than that
    // would read past the end of the embedding table; reject it instead.
    const MAX_TOKENS: usize = 512;
    let chunk = if no_chunk {
        if ids.len() > MAX_TOKENS {
            eprintln!(
                "error: {} ids, but the model caps at {} tokens per run. \
                 Drop --no-chunk to split at sentence boundaries.",
                ids.len(),
                MAX_TOKENS
            );
            std::process::exit(2);
        }
        None
    } else {
        chunk_max.or(if ids.len() > AUTO_CHUNK_TOKENS {
            Some(DEFAULT_CHUNK)
        } else {
            None
        })
    };
    if let Some(max) = chunk {
        run_chunked(&m, &mut e, &ids, max, crossfade_ms, t_load, stream);
        return;
    }

    let n = ids.len();
    let t1 = Instant::now();
    let outs = e.run_freeing(
        &m,
        vec![
            ("input_ids".into(), tensor::T::from_i64(vec![1, n], ids)),
            ("speed".into(), tensor::T::from_f32(vec![1], vec![1.0])),
        ],
    );
    let t_run = t1.elapsed();
    eprintln!("run: {:?}", t_run);
    eprintln!("rss_peak_mb: {:.1}", peak_rss_mb());

    for o in &outs {
        let audio_s = o.len() as f64 / 24000.0;
        eprintln!("out dims={:?} len={}", o.dims, o.len());
        let f = o.f();
        let mx = f.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mn = f.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("  min={mn:.4} max={mx:.4} first={:?}", &f[..f.len().min(8)]);
        if profile {
            let wall = t_load.as_secs_f64() + t_run.as_secs_f64();
            eprintln!();
            eprintln!("=== summary ===");
            eprintln!("  load            : {:.1} ms", t_load.as_secs_f64() * 1e3);
            eprintln!("  synth           : {:.1} ms", t_run.as_secs_f64() * 1e3);
            eprintln!("  wall (load+synth): {:.1} ms", wall * 1e3);
            eprintln!("  audio           : {:.3} s", audio_s);
            eprintln!("  real-time factor: {:.2}x", audio_s / t_run.as_secs_f64());
            eprintln!("  rss loaded      : {:.1} MB", rss_mb());
            eprintln!("  rss peak        : {:.1} MB", peak_rss_mb());
            eprintln!();
        }
        write_wav("out.wav", f, 24000);
        eprintln!(
            "wrote out.wav ({} samples, {:.2}s)",
            f.len(),
            f.len() as f32 / 24000.0
        );
    }
}

/// Split `ids` at sentence boundaries and synth each chunk through the same
/// `Exec`, then concatenate with a crossfade.
///
/// The model's token 0 is BOS/EOS, 4 is `.` and 3 is `,`. A chunk is
/// `[0] + ids[a..b] + [0]`, so every chunk is a well-formed utterance.
///
/// With `stream`, each chunk's audio is written to `chunk_NNN.wav` as soon as it
/// is ready, so a caller can start playing before the rest is synthesised.
fn run_chunked(
    m: &graph::Model,
    e: &mut exec::Exec,
    ids: &[i64],
    max: usize,
    crossfade_ms: f64,
    t_load: std::time::Duration,
    stream: bool,
) {
    let chunks = split_sentences(ids, max);
    eprintln!(
        "chunked: {} chunks (max {max} tokens), crossfade {crossfade_ms} ms",
        chunks.len()
    );
    let xf = (crossfade_ms / 1000.0 * 24000.0) as usize;
    let mut acc: Vec<f32> = Vec::new();
    let mut first_audio_ms: Option<f64> = None;
    let t_all = Instant::now();
    for (ci, c) in chunks.iter().enumerate() {
        let n = c.len();
        let t1 = Instant::now();
        let outs = e.run_freeing(
            m,
            vec![
                (
                    "input_ids".into(),
                    tensor::T::from_i64(vec![1, n], c.clone()),
                ),
                ("speed".into(), tensor::T::from_f32(vec![1], vec![1.0])),
            ],
        );
        let dt = t1.elapsed().as_secs_f64() * 1e3;
        if first_audio_ms.is_none() {
            first_audio_ms = Some(t_load.as_secs_f64() * 1e3 + dt);
        }
        let f = outs[0].f();
        eprintln!(
            "  chunk {}/{}: {} tokens -> {} samples ({:.2}s) in {:.1} ms",
            ci + 1,
            chunks.len(),
            n,
            f.len(),
            f.len() as f64 / 24000.0,
            dt
        );
        if stream {
            // Emit this chunk immediately so a player can start on it.
            let path = format!("chunk_{ci:03}.wav");
            write_wav(&path, f, 24000);
            eprintln!("  streamed {path} ({:.2}s)", f.len() as f64 / 24000.0);
        }
        append_crossfade(&mut acc, f, xf);
    }
    let total = t_all.elapsed().as_secs_f64() * 1e3;
    let audio_s = acc.len() as f64 / 24000.0;
    eprintln!("rss_peak_mb: {:.1}", peak_rss_mb());
    eprintln!();
    eprintln!("=== chunked summary ===");
    eprintln!("  load             : {:.1} ms", t_load.as_secs_f64() * 1e3);
    eprintln!("  synth (all chunks): {:.1} ms", total);
    eprintln!(
        "  time to first audio: {:.1} ms",
        first_audio_ms.unwrap_or(0.0)
    );
    eprintln!("  audio            : {:.3} s", audio_s);
    eprintln!("  real-time factor : {:.2}x", audio_s / (total / 1e3));
    eprintln!("  rss peak         : {:.1} MB", peak_rss_mb());
    eprintln!();
    write_wav("out.wav", &acc, 24000);
    eprintln!("wrote out.wav ({} samples, {:.2}s)", acc.len(), audio_s);
}

/// Split at `.` (token 4) only, keeping each chunk at or below `max` tokens.
///
/// A comma is a pause inside a sentence, not a sentence boundary, so splitting
/// there breaks the pacing. Commas are used only as a fallback when a single
/// sentence is longer than `max`, and a hard split is the last resort.
fn split_sentences(ids: &[i64], max: usize) -> Vec<Vec<i64>> {
    // Strip the outer BOS/EOS; each chunk gets its own.
    let body: Vec<i64> = ids.iter().copied().filter(|&t| t != 0).collect();
    // First pass: split at periods.
    let mut sentences: Vec<Vec<i64>> = Vec::new();
    let mut cur: Vec<i64> = Vec::new();
    for &t in &body {
        cur.push(t);
        if t == 4 {
            sentences.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        sentences.push(cur);
    }
    // Second pass: a sentence longer than `max` is split at commas, then hard.
    let mut chunks: Vec<Vec<i64>> = Vec::new();
    for s in sentences {
        if s.len() <= max {
            chunks.push(s);
            continue;
        }
        let mut part: Vec<i64> = Vec::new();
        for &t in &s {
            part.push(t);
            if (t == 3 && part.len() >= max / 2) || part.len() >= max {
                chunks.push(std::mem::take(&mut part));
            }
        }
        if !part.is_empty() {
            chunks.push(part);
        }
    }
    chunks
        .into_iter()
        .map(|mut c| {
            let mut v = Vec::with_capacity(c.len() + 2);
            v.push(0);
            v.append(&mut c);
            v.push(0);
            v
        })
        .collect()
}

/// Append `b` to `a`, crossfading the last `xf` samples of `a` with the first
/// `xf` of `b` so the join has no click.
fn append_crossfade(a: &mut Vec<f32>, b: &[f32], xf: usize) {
    if a.is_empty() || xf == 0 || b.len() < xf || a.len() < xf {
        a.extend_from_slice(b);
        return;
    }
    let n = a.len();
    for i in 0..xf {
        let w = i as f32 / xf as f32;
        a[n - xf + i] = a[n - xf + i] * (1.0 - w) + b[i] * w;
    }
    a.extend_from_slice(&b[xf..]);
}

// --- Windows RSS ----------------------------------------------------------

#[cfg(windows)]
mod mem {
    use std::ffi::c_void;

    #[repr(C)]
    struct Pmc {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(process: *mut c_void, counters: *mut Pmc, cb: u32) -> i32;
    }

    fn counters() -> Pmc {
        let mut c = Pmc {
            cb: std::mem::size_of::<Pmc>() as u32,
            page_fault_count: 0,
            peak_working_set_size: 0,
            working_set_size: 0,
            quota_peak_paged_pool_usage: 0,
            quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0,
            quota_non_paged_pool_usage: 0,
            pagefile_usage: 0,
            peak_pagefile_usage: 0,
            private_usage: 0,
        };
        unsafe {
            K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb);
        }
        c
    }

    pub fn rss_mb() -> f64 {
        counters().working_set_size as f64 / 1e6
    }

    pub fn peak_rss_mb() -> f64 {
        counters().peak_working_set_size as f64 / 1e6
    }
}

#[cfg(windows)]
use mem::{peak_rss_mb, rss_mb};

#[cfg(not(windows))]
fn rss_mb() -> f64 {
    0.0
}
#[cfg(not(windows))]
fn peak_rss_mb() -> f64 {
    0.0
}

fn write_wav(path: &str, samples: &[f32], rate: u32) {
    let n = samples.len() as u32;
    let data_len = n * 2;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b).expect("write wav");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model caps at 512 tokens per run. `split_sentences` must never
    /// produce a chunk whose body exceeds `max`, because the caller adds the
    /// BOS/EOS pair on top.
    #[test]
    fn split_sentences_respects_max() {
        for n in [510usize, 511, 512, 513, 1024, 2000] {
            // One long sentence: no periods, no commas, so only the hard split
            // can bound it.
            let ids: Vec<i64> = std::iter::once(0)
                .chain(std::iter::repeat_n(16, n - 2))
                .chain(std::iter::once(0))
                .collect();
            let chunks = split_sentences(&ids, 100);
            assert!(!chunks.is_empty(), "n={n}: no chunks");
            for c in &chunks {
                assert!(
                    c.len() <= 102,
                    "n={n}: chunk of {} ids exceeds max 100 + 2 BOS/EOS",
                    c.len()
                );
                assert_eq!(c[0], 0, "n={n}: chunk must start with BOS");
                assert_eq!(*c.last().unwrap(), 0, "n={n}: chunk must end with EOS");
            }
            // Every id survives the round trip (BOS/EOS stripped and re-added).
            let total: usize = chunks.iter().map(|c| c.len() - 2).sum();
            assert_eq!(total, n - 2, "n={n}: ids lost in splitting");
        }
    }

    /// Sentence and comma boundaries produce well-formed chunks.
    #[test]
    fn split_sentences_at_punctuation() {
        // "Hello. There, world." with the vocab's ids for . , and space.
        let ids = vec![0, 50, 4, 16, 50, 3, 16, 50, 4, 0];
        let chunks = split_sentences(&ids, 100);
        assert_eq!(chunks.len(), 2, "two sentences -> two chunks");
        for c in &chunks {
            assert_eq!(c[0], 0);
            assert_eq!(*c.last().unwrap(), 0);
        }
    }
}
