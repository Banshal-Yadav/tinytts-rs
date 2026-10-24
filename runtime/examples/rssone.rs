//! One fresh process = one run. Prints loaded RSS, peak LIVE (counting
//! allocator) and peak RSS, so the RSS high-water mark is never contaminated by
//! an earlier run in the same process.
//!
//! Run:  cargo run --release --example rssone -- <tokens...>

use std::time::Instant;
use tinytts::alloc;
use tinytts::{exec::Exec, graph, onnx, tensor::T};

#[global_allocator]
static A: alloc::Counting = alloc::Counting;

#[cfg(windows)]
mod mem {
    use std::ffi::c_void;
    #[repr(C)]
    struct Pmc {
        cb: u32,
        pf: u32,
        peak: usize,
        ws: usize,
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        e: usize,
        f: usize,
    }
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(p: *mut c_void, c: *mut Pmc, cb: u32) -> i32;
    }
    fn c() -> Pmc {
        let mut c = Pmc {
            cb: std::mem::size_of::<Pmc>() as u32,
            pf: 0,
            peak: 0,
            ws: 0,
            a: 0,
            b: 0,
            c: 0,
            d: 0,
            e: 0,
            f: 0,
        };
        unsafe {
            K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb);
        }
        c
    }
    pub fn rss() -> f64 {
        c().ws as f64 / 1e6
    }
    pub fn peak() -> f64 {
        c().peak as f64 / 1e6
    }
}

fn main() {
    let ids: Vec<i64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let n = ids.len();

    let model = std::env::var("MODEL").unwrap_or_else(|_| "../paradee_int8.onnx".to_string());
    let g = {
        let bytes = std::fs::read(&model).unwrap_or_else(|e| panic!("read {model}: {e}"));
        onnx::parse(&bytes)
    };
    let m = graph::Model::new(g);
    let mut e = Exec::new();
    e.load(
        &m,
        vec![
            ("input_ids".into(), T::from_i64(vec![1, n], ids)),
            ("speed".into(), T::from_f32(vec![1], vec![1.0])),
        ],
    );
    let loaded = mem::rss();

    alloc::reset();
    let t0 = Instant::now();
    let outs = e.run_freeing(
        &m,
        vec![
            ("input_ids".into(), T::from_i64(vec![1, n], vec![0; n])),
            ("speed".into(), T::from_f32(vec![1], vec![1.0])),
        ],
    );
    let synth = t0.elapsed().as_secs_f64() * 1e3;

    println!("tokens={n}");
    println!("loaded_rss_mb={loaded:.1}");
    println!("peak_live_mb={:.1}", alloc::peak_mb());
    println!("peak_rss_mb={:.1}", mem::peak());
    println!("synth_ms={synth:.1}");
    println!("out_len={}", outs[0].len());
}
