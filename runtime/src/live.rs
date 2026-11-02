//! tinytts-live: true live streaming TTS in the terminal.
//!
//! Loads model + G2P once, reads lines from stdin, synthesizes in chunks
//! with overlapping synthesis/playback via a ring buffer, streams audio
//! directly to the default output device via cpal.

use std::io::{self, BufRead};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

const SAMPLE_RATE: u32 = 24000;
const CHUNK_MAX_TOKENS: usize = 80;
const CROSSFADE_SAMPLES: usize = 240;
// Interleaved device-rate samples the producer may run ahead of playback.
// ~10.9 s at 48 kHz stereo: enough to absorb a whole burst of chunks, so the
// writer is never in a race with the audio callback. (Was 48000 = 0.68 s,
// which forced the producer to track the device sample by sample.)
const RING_BUFFER_CAPACITY: usize = 1 << 20;
// Time for the device to play out whatever it already copied out of the ring.
const DEVICE_DRAIN_MS: u64 = 300;
// Give up on playback if the device hasn't consumed a single sample for this long.
const STALL_ABORT_MS: u64 = 3000;

struct RingBuffer {
    buffer: *mut f32,
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
    underruns: AtomicUsize,
}

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two();
        let mut v = vec![0.0f32; cap];
        let ptr = v.as_mut_ptr();
        std::mem::forget(v);
        Self { buffer: ptr, capacity: cap, head: AtomicUsize::new(0), tail: AtomicUsize::new(0), underruns: AtomicUsize::new(0) }
    }
    fn underrun_count(&self) -> usize { self.underruns.load(Ordering::Relaxed) }
    #[inline] fn mask(&self) -> usize { self.capacity - 1 }
    #[inline] fn available_read(&self) -> usize {
        let h = self.head.load(Ordering::Acquire);
        let t = self.tail.load(Ordering::Acquire);
        (h.wrapping_sub(t)) & self.mask()
    }
    fn try_write(&self, src: &[f32]) -> usize {
        let h = self.head.load(Ordering::Acquire);
        let t = self.tail.load(Ordering::Acquire);
        let free = self.capacity - 1 - (h.wrapping_sub(t)) & self.mask();
        let to_write = src.len().min(free);
        if to_write == 0 { return 0; }
        let mask = self.mask();
        let first = (self.capacity - (h & mask)).min(to_write);
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.buffer.add(h & mask), first);
            if first < to_write {
                std::ptr::copy_nonoverlapping(src.as_ptr().add(first), self.buffer, to_write - first);
            }
        }
        self.head.store(h.wrapping_add(to_write), Ordering::Release);
        to_write
    }
    /// Producer side. Blocks while the ring is full, but never indefinitely:
    /// if playback stops consuming (dead stream / unplugged device) it gives up
    /// after `STALL_ABORT_MS` and reports, instead of hanging the terminal.
    fn write(&self, src: &[f32], stop: &AtomicBool) -> usize {
        let mut w = 0;
        let mut last_tail = self.tail.load(Ordering::Acquire);
        let mut idle_ms = 0u32;
        while w < src.len() && !stop.load(Ordering::Acquire) {
            let n = self.try_write(&src[w..]);
            if n > 0 { w += n; idle_ms = 0; last_tail = self.tail.load(Ordering::Acquire); continue; }
            thread::sleep(Duration::from_millis(1));
            let t = self.tail.load(Ordering::Acquire);
            if t == last_tail {
                idle_ms += 1;
                if idle_ms >= STALL_ABORT_MS as u32 {
                    eprintln!("[live] WARN: playback stalled, dropping {} samples", src.len() - w);
                    return w;
                }
            } else { idle_ms = 0; last_tail = t; }
        }
        w
    }
    fn try_read(&self, dst: &mut [f32]) -> usize {
        let h = self.head.load(Ordering::Acquire);
        let t = self.tail.load(Ordering::Acquire);
        let avail = (h.wrapping_sub(t)) & self.mask();
        let to_read = dst.len().min(avail);
        if to_read == 0 { return 0; }
        let mask = self.mask();
        let first = (self.capacity - (t & mask)).min(to_read);
        unsafe {
            std::ptr::copy_nonoverlapping(self.buffer.add(t & mask), dst.as_mut_ptr(), first);
            if first < to_read {
                std::ptr::copy_nonoverlapping(self.buffer, dst.as_mut_ptr().add(first), to_read - first);
            }
        }
        self.tail.store(t.wrapping_add(to_read), Ordering::Release);
        to_read
    }
    /// Audio-callback path. MUST NOT BLOCK: an audio callback that misses its
    /// deadline gets the stream torn down by the driver, which is why playback
    /// used to die at a random word. Underruns become silence, not a stall.
    fn read(&self, dst: &mut [f32]) -> usize {
        let n = self.try_read(dst);
        if n < dst.len() {
            dst[n..].fill(0.0);
            self.underruns.fetch_add(1, Ordering::Relaxed);
        }
        n
    }
}

unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}
fn split_chunks(ids: &[i64], max: usize) -> Vec<Vec<i64>> {
    let body: Vec<i64> = ids.iter().copied().filter(|&t| t!=0).collect();
    let mut sent = Vec::new(); let mut cur = Vec::new();
    for &t in &body { cur.push(t); if t==4 { sent.push(std::mem::take(&mut cur)); } }
    if !cur.is_empty() { sent.push(cur); }
    let mut chunks = Vec::new();
    for s in sent {
        if s.len() <= max { chunks.push(s); continue; }
        let mut part = Vec::new();
        for &t in &s { part.push(t);
            if (t==3 && part.len()>=max/2) || part.len()>=max { chunks.push(std::mem::take(&mut part)); }
        }
        if !part.is_empty() { chunks.push(part); }
    }
    chunks.into_iter().map(|mut c| { let mut v=Vec::with_capacity(c.len()+2); v.push(0); v.append(&mut c); v.push(0); v }).collect()
}

fn synth_chunk(m: &graph::Model, e: &mut exec::Exec, ids: &[i64], speed: f32) -> Vec<f32> {
    let n = ids.len();
    let outs = e.run_freeing(m, vec![
        ("input_ids".into(), tensor::T::from_i64(vec![1,n], ids.to_vec())),
        ("speed".into(), tensor::T::from_f32(vec![1], vec![speed])),
    ]);
    outs[0].f().to_vec()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).expect("usage: tinytts-live <model.onnx> [--chunk N] [--device NAME] [--speed F]");
    let mut chunk_max = CHUNK_MAX_TOKENS; let mut dev = None; let mut speed = 1.0f32; let mut i=2;
    while i < args.len() {
        match args[i].as_str() {
            "--chunk" => { if let Some(v)=args.get(i+1).and_then(|s|s.parse().ok()) { chunk_max=v; } i+=2; }
            "--device" => { dev=args.get(i+1).cloned(); i+=2; }
            "--speed" => { if let Some(v)=args.get(i+1).and_then(|s|s.parse().ok()) { speed=v; } i+=2; }
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
    let mut config = device.supported_output_configs().unwrap()
        .find(|c| c.channels() == 1 && c.min_sample_rate().0 <= SAMPLE_RATE && c.max_sample_rate().0 >= SAMPLE_RATE)
        .map(|c| c.with_sample_rate(cpal::SampleRate(SAMPLE_RATE)))
        .or_else(|| device.default_output_config().ok())
        .expect("no supported config");
    let channels = config.channels();
    let device_rate = config.sample_rate().0;
    let need_resample = device_rate != SAMPLE_RATE;
    let need_interleave = channels == 2;
    eprintln!("[live] audio config: {:?} (resample={}, interleave={})", config, need_resample, need_interleave);
    eprintln!("[live] rss after load: {:.1} MB", rss_mb());

    /// Linear resample from 24kHz to device rate
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

    /// Duplicate each sample for stereo (L=R=sample)
    fn interleave_stereo(src: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(src.len() * 2);
        for &s in src { out.push(s); out.push(s); }
        out
    }

    let ring = Arc::new(RingBuffer::new(RING_BUFFER_CAPACITY));
    let stop = Arc::new(AtomicBool::new(false));
    let rc = ring.clone();
    let stream = device.build_output_stream(&config.into(),
        move |data: &mut [f32], _| { rc.read(data); },
        |e| eprintln!("[live] stream error: {e}"), None).expect("stream");
    stream.play().expect("play");
    eprintln!("[live] ready. Type sentences and press Enter. Ctrl+C to quit.");
    let stdin = io::stdin(); let mut line_num=0;
    for line in stdin.lock().lines() {
        let line = match line { Ok(l)=>l.trim().to_string(), Err(e)=>{ eprintln!("[live] stdin error: {e}"); break; } };
        if line.is_empty() { continue; }
        line_num+=1;
        let t_g2p = Instant::now(); let (ids, oov) = g2p_ids(&line); let t_g2p_ms = t_g2p.elapsed().as_secs_f64()*1e3;
        if !oov.is_empty() { eprintln!("[live] OOV: {}", oov.join(", ")); }
        let chunks = split_chunks(&ids, chunk_max);
        if chunks.is_empty() { eprintln!("[live] no tokens for line {line_num}"); continue; }
        let t_first = Instant::now(); let mut first=false; let mut tot_synth=0.0; let mut tot_samp=0usize;
        // Tail of the previous chunk, held back so the next chunk's head can be
        // overlap-added onto it. (The old code blended the *new* chunk's tail
        // with the *old* chunk's head, which is not a crossfade at all.)
        let mut pending: Option<Vec<f32>> = None;
        for (ci, ch) in chunks.iter().enumerate() {
            let t_s = Instant::now(); let audio = synth_chunk(&m, &mut e, ch, speed); let s_ms = t_s.elapsed().as_secs_f64()*1e3;
            tot_synth+=s_ms;
            let mut ca = audio;
            if need_resample { ca = resample(&ca, SAMPLE_RATE, device_rate); }
            if need_interleave { ca = interleave_stereo(&ca); }
            tot_samp += ca.len();
            let hl = CROSSFADE_SAMPLES.min(ca.len() / 2);
            if let Some(t) = pending.take() {
                let n = hl.min(t.len());
                for i in 0..n { let w = i as f32 / n as f32; ca[i] = t[i] * (1.0 - w) + ca[i] * w; }
            }
            let end = ca.len() - hl;
            ring.write(&ca[..end], &stop);
            pending = Some(ca[end..].to_vec());
            if !first { let tf = t_first.elapsed().as_secs_f64()*1e3; eprintln!("[live] line {line_num}: first audio at {tf:.1} ms (chunk {}/{})", ci+1, chunks.len()); first=true; }
            let mono_ca = ca.len() / channels as usize;
            eprintln!("[live]   chunk {}/{}: {} tokens -> {} frames ({:.2}s) in {:.1} ms", ci+1, chunks.len(), ch.len(), ca.len(), mono_ca as f64/device_rate as f64, s_ms);
        }
        if let Some(t) = pending.take() { ring.write(&t, &stop); }
        // Wait for playback to consume the queue, then give the device time to
        // play out what it already copied (available_read()==0 means "copied",
        // not "heard"). Bail out with a warning if the stream has died.
        let mut last = ring.available_read(); let mut idle = 0u32;
        while ring.available_read() > 0 && !stop.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(10));
            let now = ring.available_read();
            if now >= last {
                idle += 1;
                if idle as u64 >= STALL_ABORT_MS / 10 { eprintln!("[live] WARN: playback stalled with {now} samples queued"); break; }
            } else { idle = 0; }
            last = now;
        }
        thread::sleep(Duration::from_millis(DEVICE_DRAIN_MS));
        // tot_samp is interleaved frames at device_rate; mono frames = tot_samp / channels
        // duration = mono_frames / device_rate (since resample already happened)
        let mono_frames = tot_samp / channels as usize;
        let dur = mono_frames as f64 / device_rate as f64;
        let rtf = dur / (tot_synth / 1000.0);
        eprintln!("[live] line {line_num} done: g2p={t_g2p_ms:.1}ms synth={tot_synth:.1}ms audio={dur:.2}s RTF={rtf:.2}x rss={:.1}MB peak={:.1}MB underruns={}", rss_mb(), peak_rss_mb(), ring.underrun_count());
    }
    stop.store(true, Ordering::Release); stream.pause().ok(); 
    eprintln!("[live] stopped rss={:.1}MB peak={:.1}MB", rss_mb(), peak_rss_mb());
}
