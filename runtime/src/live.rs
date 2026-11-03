//! tinytts-live: true live streaming TTS in the terminal.
//!
//! Loads model + G2P once, reads lines from stdin, synthesizes each line into a
//! complete PCM buffer, then streams that buffer to the default output device
//! via cpal.
//!
//! Playback used to run through a producer/consumer ring while the next chunk
//! was still being synthesized. That is where the audio died: the callback
//! could be handed an empty ring and WASAPI inserted silence. A line is now
//! fully rendered before the device is opened, so the callback copies from a
//! complete buffer and cannot starve. See docs/live_playback_stall.md.

use std::io::{self, BufRead};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use tinytts::{exec, graph, onnx, tensor};
use tinytts_g2p::g2p_ids;

#[cfg(windows)]
mod mem {
    use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use winapi::um::processthreadsapi::GetCurrentProcess;
    fn counters() -> PROCESS_MEMORY_COUNTERS {
        let mut c = PROCESS_MEMORY_COUNTERS {
            cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            PageFaultCount: 0,
            PeakWorkingSetSize: 0,
            WorkingSetSize: 0,
            QuotaPeakPagedPoolUsage: 0,
            QuotaPagedPoolUsage: 0,
            QuotaPeakNonPagedPoolUsage: 0,
            QuotaNonPagedPoolUsage: 0,
            PagefileUsage: 0,
            PeakPagefileUsage: 0,
        };
        unsafe {
            GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb);
        }
        c
    }
    pub fn rss_mb() -> f64 { counters().WorkingSetSize as f64 / 1e6 }
    pub fn peak_rss_mb() -> f64 { counters().PeakWorkingSetSize as f64 / 1e6 }
}

#[cfg(not(windows))]
mod mem {
    pub fn rss_mb() -> f64 { 0.0 }
    pub fn peak_rss_mb() -> f64 { 0.0 }
}

use mem::{rss_mb, peak_rss_mb};

/// The cpal callback runs on a WASAPI-owned thread with normal priority. Any
/// scheduling hiccup there makes the engine run dry, and WASAPI answers with
/// silence. Raise it once, from inside the callback, so cpal's thread is the
/// one that gets the CPU.
#[cfg(windows)]
fn boost_audio_thread() {
    static DONE: AtomicBool = AtomicBool::new(false);
    if DONE.swap(true, Ordering::Relaxed) { return; }
    unsafe {
        use winapi::um::processthreadsapi::{GetCurrentThread, SetThreadPriority};
        use winapi::um::winbase::THREAD_PRIORITY_TIME_CRITICAL;
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL as i32);
    }
}

#[cfg(not(windows))]
fn boost_audio_thread() {}

const SAMPLE_RATE: u32 = 24000;
// Content tokens per chunk. Small enough that one synth call is a short CPU
// burst (~2.5 s of audio) and a long sentence never becomes one huge run.
const CHUNK_MAX_TOKENS: usize = 40;
// Hard ceiling: the model's embedding table is 512 slots, BOS/EOS included.
// 600 tokens overruns it ("range end index 4104 out of range for slice of
// length 4096"). 510 content tokens is the largest verified-good input.
const MODEL_MAX_TOKENS: usize = 510;
// Don't leave a fragment shorter than this at the end of a sentence.
const CHUNK_MIN_TOKENS: usize = 10;
const CROSSFADE_SAMPLES: usize = 240;
// Time for the device to play out what it has already buffered after the last
// sample has been handed over.
const DEVICE_DRAIN_MS: u64 = 300;

// Punctuation token ids from vocab.txt: ';'=1 ':'=2 ','=3 '.'=4 '!'=5 '?'=6.
fn is_sentence_end(t: i64) -> bool { t == 4 || t == 5 || t == 6 }
fn is_clause_end(t: i64) -> bool { t == 1 || t == 2 || t == 3 }

/// Split text into chunks of at most `max` content tokens, always at a
/// sentence end when possible, then at a clause break, and only as a last
/// resort mid-clause. No chunk may exceed `MODEL_MAX_TOKENS`.
fn split_chunks(ids: &[i64], max: usize) -> Vec<Vec<i64>> {
    let max = max.clamp(CHUNK_MIN_TOKENS, MODEL_MAX_TOKENS);
    let body: Vec<i64> = ids.iter().copied().filter(|&t| t != 0).collect();

    let mut sent: Vec<Vec<i64>> = Vec::new();
    let mut cur = Vec::new();
    for &t in &body {
        cur.push(t);
        if is_sentence_end(t) { sent.push(std::mem::take(&mut cur)); }
    }
    if !cur.is_empty() { sent.push(cur); }

    let mut chunks: Vec<Vec<i64>> = Vec::new();
    for s in sent {
        let mut rest = &s[..];
        while rest.len() > max {
            let mut cut = max;
            for i in (max / 3..max).rev() {
                if is_clause_end(rest[i]) { cut = i + 1; break; }
            }
            chunks.push(rest[..cut].to_vec());
            rest = &rest[cut..];
        }
        if !rest.is_empty() { chunks.push(rest.to_vec()); }
    }

    // Absorb runt fragments so we don't synthesise 4-token slivers.
    let mut merged: Vec<Vec<i64>> = Vec::new();
    for c in chunks {
        if let Some(last) = merged.last_mut() {
            let runt = c.len() < CHUNK_MIN_TOKENS || last.len() < CHUNK_MIN_TOKENS;
            if runt && last.len() + c.len() <= max { last.extend_from_slice(&c); continue; }
        }
        merged.push(c);
    }

    merged.into_iter().map(|mut c| { let mut v = Vec::with_capacity(c.len() + 2); v.push(0); v.append(&mut c); v.push(0); v }).collect()
}

fn synth_chunk(m: &graph::Model, e: &mut exec::Exec, ids: &[i64], speed: f32) -> Vec<f32> {
    let n = ids.len();
    let outs = e.run_freeing(m, vec![
        ("input_ids".into(), tensor::T::from_i64(vec![1,n], ids.to_vec())),
        ("speed".into(), tensor::T::from_f32(vec![1], vec![speed])),
    ]);
    outs[0].f().to_vec()
}

/// Linear resample from 24 kHz to the device rate.
fn resample(src: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to { return src.to_vec(); }
    let ratio = from as f32 / to as f32;
    let out_len = (src.len() as f32 / ratio).ceil() as usize;
    let mut out = vec![0.0f32; out_len];
    for i in 0..out_len {
        let src_pos = i as f32 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f32;
        if idx + 1 < src.len() {
            out[i] = src[idx] * (1.0 - frac) + src[idx + 1] * frac;
        } else if idx < src.len() {
            out[i] = src[idx];
        }
    }
    out
}

/// Duplicate each sample for stereo (L=R=sample).
fn interleave_stereo(src: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for &s in src { out.push(s); out.push(s); }
    out
}

/// Record the default input device for `secs` and write it as a mono WAV at the
/// device rate. Diagnostics only: answers "did the speaker actually make this
/// sound?" for a session where the software path already reported success.
fn record_to_wav(path: String, secs: f64) {
    let host = cpal::default_host();
    let dev = match host.default_input_device() {
        Some(d) => d,
        None => { eprintln!("[live] rec: no input device"); return; }
    };
    let cfg = match dev.default_input_config() {
        Ok(c) => c,
        Err(e) => { eprintln!("[live] rec: {e}"); return; }
    };
    let rate = cfg.sample_rate().0;
    let ch = cfg.channels() as usize;
    let stop = Arc::new(AtomicBool::new(false));
    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    match cfg.sample_format() {
        cpal::SampleFormat::F32 => {
            let (b, s) = (buf.clone(), stop.clone());
            let st = dev.build_input_stream(&cfg.clone().into(), move |d: &[f32], _| {
                if !s.load(Ordering::Relaxed) { let mut v = b.lock().unwrap(); v.extend((0..d.len()).step_by(ch).map(|i| d[i])); }
            }, |e| eprintln!("[live] rec error: {e}"), None);
            let st = match st { Ok(s) => s, Err(e) => { eprintln!("[live] rec: {e}"); return; } };
            st.play().ok();
            thread::sleep(Duration::from_secs_f64(secs));
            stop.store(true, Ordering::Relaxed);
            drop(st);
        }
        cpal::SampleFormat::I16 => {
            let (b, s) = (buf.clone(), stop.clone());
            let st = dev.build_input_stream(&cfg.clone().into(), move |d: &[i16], _| {
                if !s.load(Ordering::Relaxed) { let mut v = b.lock().unwrap(); v.extend((0..d.len()).step_by(ch).map(|i| d[i] as f32 / 32768.0)); }
            }, |e| eprintln!("[live] rec error: {e}"), None);
            let st = match st { Ok(s) => s, Err(e) => { eprintln!("[live] rec: {e}"); return; } };
            st.play().ok();
            thread::sleep(Duration::from_secs_f64(secs));
            stop.store(true, Ordering::Relaxed);
            drop(st);
        }
        f => { eprintln!("[live] rec: unsupported format {f:?}"); return; }
    }
    let v = buf.lock().unwrap();
    write_wav(&path, &v, rate);
    eprintln!("[live] rec: {} samples at {} Hz ({:.1}s)", v.len(), rate, v.len() as f64 / rate as f64);
}

fn write_wav(path: &str, samples: &[f32], rate: u32) {
    let data_len = (samples.len() * 2) as u32;
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).expect("usage: tinytts-live <model.onnx> [--chunk N] [--device NAME] [--speed F] [--wav PATH] [--rec PATH]");
    let mut chunk_max = CHUNK_MAX_TOKENS; let mut dev = None; let mut speed = 1.0f32;
    let mut wav: Option<String> = None; let mut rec: Option<String> = None; let mut i=2;
    while i < args.len() {
        match args[i].as_str() {
            "--chunk" => { if let Some(v)=args.get(i+1).and_then(|s|s.parse().ok()) { chunk_max=v; } i+=2; }
            "--device" => { dev=args.get(i+1).cloned(); i+=2; }
            "--speed" => { if let Some(v)=args.get(i+1).and_then(|s|s.parse().ok()) { speed=v; } i+=2; }
            "--wav" => { wav=args.get(i+1).cloned(); i+=2; }
            "--rec" => { rec=args.get(i+1).cloned(); i+=2; }
            _ => i+=1,
        }
    }
    let t0 = Instant::now();
    let g = onnx::parse(&std::fs::read(model).expect("read model"));
    let m = graph::Model::new(g);
    eprintln!("[live] model loaded in {:.1} ms (nodes={}, inits={})", t0.elapsed().as_secs_f64()*1e3, m.g.nodes.len(), m.g.inits.len());
    let mut e = exec::Exec::new(); e.free_dead = true;
    let _ = synth_chunk(&m, &mut e, &[0,16,43,44,45,16,4,0], 1.0);
    eprintln!("[live] warm-up done");
    eprintln!("[live] rss after warmup: {:.1} MB (peak: {:.1} MB)", rss_mb(), peak_rss_mb());
    let host = cpal::default_host();
    let device = if let Some(n)=dev { host.output_devices().unwrap().find(|d|d.name().ok().as_deref()==Some(&n)).expect("device") }
        else { host.default_output_device().expect("no device") };
    eprintln!("[live] audio device: {}", device.name().unwrap_or_default());
    // Request 24000Hz mono config; if unavailable, use stereo and interleave
    let config = device.supported_output_configs().unwrap()
        .find(|c| c.channels() == 1 && c.min_sample_rate().0 <= SAMPLE_RATE && c.max_sample_rate().0 >= SAMPLE_RATE)
        .map(|c| c.with_sample_rate(cpal::SampleRate(SAMPLE_RATE)))
        .or_else(|| device.default_output_config().ok())
        .expect("no supported config");
    let channels = config.channels() as usize;
    let device_rate = config.sample_rate().0;
    let need_resample = device_rate != SAMPLE_RATE;
    let need_interleave = channels == 2;
    eprintln!("[live] audio config: {:?} (resample={}, interleave={})", config, need_resample, need_interleave);
    eprintln!("[live] rss after load: {:.1} MB", rss_mb());
    eprintln!("[live] ready. Type sentences and press Enter. Ctrl+C to quit.");

    let stdin = io::stdin(); let mut line_num=0;
    let mut recs: Vec<thread::JoinHandle<()>> = Vec::new();
    for line in stdin.lock().lines() {
        let line = match line { Ok(l)=>l.trim().to_string(), Err(e)=>{ eprintln!("[live] stdin error: {e}"); break; } };
        if line.is_empty() { continue; }
        line_num+=1;
        let t_g2p = Instant::now(); let (ids, oov) = g2p_ids(&line); let t_g2p_ms = t_g2p.elapsed().as_secs_f64()*1e3;
        if !oov.is_empty() { eprintln!("[live] OOV: {}", oov.join(", ")); }
        let chunks = split_chunks(&ids, chunk_max);
        if chunks.is_empty() { eprintln!("[live] no tokens for line {line_num}"); continue; }

        // 1. Render the whole line into one 24 kHz mono buffer. Chunk N's tail
        //    is held back and overlap-added into chunk N+1's head.
        let t_synth = Instant::now();
        let mut mono: Vec<f32> = Vec::new();
        let mut pending: Option<Vec<f32>> = None;
        for (ci, ch) in chunks.iter().enumerate() {
            let t_s = Instant::now();
            let mut a = synth_chunk(&m, &mut e, ch, speed);
            let s_ms = t_s.elapsed().as_secs_f64()*1e3;
            let hl = CROSSFADE_SAMPLES.min(a.len() / 2);
            if let Some(t) = pending.take() {
                let n = hl.min(t.len());
                for i in 0..n { let w = i as f32 / n as f32; a[i] = t[i] * (1.0 - w) + a[i] * w; }
            }
            let end = a.len() - hl;
            mono.extend_from_slice(&a[..end]);
            pending = Some(a[end..].to_vec());
            eprintln!("[live]   chunk {}/{}: {} tokens -> {} frames ({:.2}s) in {:.1} ms",
                ci+1, chunks.len(), ch.len(), a.len(), a.len() as f64 / SAMPLE_RATE as f64, s_ms);
        }
        if let Some(t) = pending.take() { mono.extend_from_slice(&t); }
        let synth_ms = t_synth.elapsed().as_secs_f64()*1e3;
        let dur = mono.len() as f64 / SAMPLE_RATE as f64;

        // 2. Resample/interleave once, so the device gets one contiguous buffer.
        if let Some(p) = &wav { write_wav(p, &mono, SAMPLE_RATE); }
        let resampled = if need_resample { Some(resample(&mono, SAMPLE_RATE, device_rate)) } else { None };
        let mono = resampled.unwrap_or(mono);
        let pcm = if need_interleave { interleave_stereo(&mono) } else { mono };

        if let Some(rp) = &rec {
            let rp = rp.clone(); let secs = dur + 2.5;
            recs.push(thread::spawn(move || record_to_wav(rp, secs)));
        }

        // 3. Play it. The callback copies from a buffer that is already complete,
        //    so it cannot run out of data. It can still be starved of CPU, which
        //    is what the gap counters below measure: if the engine asks for the
        //    next block later than the previous block lasts, it played silence.
        let total = pcm.len();
        let pcm = Arc::new(pcm);
        let pos = Arc::new(AtomicUsize::new(0));
        let base_t = Instant::now();
        let last_us = Arc::new(AtomicU64::new(0));
        let gap_max_us = Arc::new(AtomicU64::new(0));
        let cbs = Arc::new(AtomicU64::new(0));
        let starved = Arc::new(AtomicU64::new(0));
        let block_frames = Arc::new(AtomicU64::new(0));
        let (d2, p2) = (pcm.clone(), pos.clone());
        let (lu, gmu, cb, st, bf) = (last_us.clone(), gap_max_us.clone(), cbs.clone(), starved.clone(), block_frames.clone());
        let stream = device.build_output_stream(&config.clone().into(),
            move |out: &mut [f32], _| {
                boost_audio_thread();
                let now = base_t.elapsed().as_micros() as u64;
                let prev = lu.swap(now, Ordering::Relaxed);
                if prev != 0 {
                    let gap = now - prev;
                    gmu.fetch_max(gap, Ordering::Relaxed);
                    let frames = out.len() / channels;
                    let last = bf.load(Ordering::Relaxed);
                    if last != 0 && gap > last.saturating_mul(3) * 1_000_000 / device_rate as u64 { st.fetch_add(1, Ordering::Relaxed); }
                    bf.store(frames as u64, Ordering::Relaxed);
                }
                cb.fetch_add(1, Ordering::Relaxed);
                let p = p2.load(Ordering::Relaxed);
                let n = out.len().min(d2.len().saturating_sub(p));
                out[..n].copy_from_slice(&d2[p..p + n]);
                if n < out.len() { out[n..].fill(0.0); }
                p2.store(p + n, Ordering::Relaxed);
            },
            |err| eprintln!("[live] stream error: {err}"), None).expect("stream");
        stream.play().expect("play");
        let t_play = Instant::now();
        eprintln!("[live] line {line_num}: {} chunks, {:.2}s of audio synthesised in {:.1} ms (RTF={:.2}x), playing",
            chunks.len(), dur, synth_ms, dur / (synth_ms / 1000.0));

        let mut last = 0usize; let mut idle = 0u32;
        loop {
            let p = pos.load(Ordering::Relaxed);
            if p >= total { break; }
            thread::sleep(Duration::from_millis(20));
            if p == last {
                idle += 1;
                if idle >= 150 { eprintln!("[live] WARN: playback stalled at {p}/{total} samples"); break; }
            } else { idle = 0; last = p; }
        }
        thread::sleep(Duration::from_millis(DEVICE_DRAIN_MS));
        stream.pause().ok();
        drop(stream);
        let play_s = t_play.elapsed().as_secs_f64();
        let (ncb, gap, nst) = (cbs.load(Ordering::Relaxed), gap_max_us.load(Ordering::Relaxed), starved.load(Ordering::Relaxed));
        let blk = block_frames.load(Ordering::Relaxed);
        eprintln!("[live] line {line_num} done: g2p={t_g2p_ms:.1}ms synth={synth_ms:.1}ms audio={dur:.2}s RTF={:.2}x play={play_s:.2}s block={:.1}ms cb={ncb} gap_max={:.1}ms starved={nst} rss={:.1}MB peak={:.1}MB",
            dur / (synth_ms / 1000.0), blk as f64 * 1e3 / device_rate as f64, gap as f64 / 1000.0, rss_mb(), peak_rss_mb());
    }
    for h in recs { let _ = h.join(); }
    eprintln!("[live] stopped rss={:.1}MB peak={:.1}MB", rss_mb(), peak_rss_mb());
}
