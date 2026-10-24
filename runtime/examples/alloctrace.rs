//! Allocation counts and the large-allocation total for one run.
//!
//! Run:  cargo run --release --example alloctrace -- <tokens...>

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use tinytts::{exec::Exec, graph, onnx, tensor::T};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static N: AtomicUsize = AtomicUsize::new(0);
static BIG: AtomicUsize = AtomicUsize::new(0);
static BIG_N: AtomicUsize = AtomicUsize::new(0);

struct A;

unsafe impl GlobalAlloc for A {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let live = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
            N.fetch_add(1, Ordering::Relaxed);
            if l.size() >= 262144 {
                BIG.fetch_add(l.size(), Ordering::Relaxed);
                BIG_N.fetch_add(1, Ordering::Relaxed);
            }
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        let cur = LIVE.load(Ordering::Relaxed);
        LIVE.store(cur.saturating_sub(l.size()), Ordering::Relaxed);
        System.dealloc(p, l);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = System.realloc(p, l, new);
        if !q.is_null() {
            LIVE.fetch_sub(l.size(), Ordering::Relaxed);
            let live = LIVE.fetch_add(new, Ordering::Relaxed) + new;
            PEAK.fetch_max(live, Ordering::Relaxed);
            N.fetch_add(1, Ordering::Relaxed);
            if new >= 262144 {
                BIG.fetch_add(new, Ordering::Relaxed);
                BIG_N.fetch_add(1, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static GA: A = A;

fn main() {
    let ids: Vec<i64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let n = ids.len();
    let path = std::env::var("MODEL").unwrap_or_else(|_| "../paradee_int8.onnx".to_string());
    let g = {
        let bytes = std::fs::read(&path).unwrap();
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
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    N.store(0, Ordering::Relaxed);
    BIG.store(0, Ordering::Relaxed);
    BIG_N.store(0, Ordering::Relaxed);
    let _ = e.run_freeing(
        &m,
        vec![
            ("input_ids".into(), T::from_i64(vec![1, n], vec![0; n])),
            ("speed".into(), T::from_f32(vec![1], vec![1.0])),
        ],
    );
    println!("tokens={n}");
    println!("allocs={}", N.load(Ordering::Relaxed));
    println!(
        "peak_live_mb={:.1}",
        PEAK.load(Ordering::Relaxed) as f64 / 1e6
    );
    println!(
        "allocs_ge_256kb: n={} total_mb={:.1}",
        BIG_N.load(Ordering::Relaxed),
        BIG.load(Ordering::Relaxed) as f64 / 1e6
    );
}
