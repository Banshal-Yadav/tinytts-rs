//! A2: a counting global allocator, so we can measure peak LIVE bytes (what the
//! program actually holds) separately from peak RSS (what the OS reports).
//!
//! Enable with `--features counting-alloc`. Without it this is a no-op.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub static LIVE: AtomicUsize = AtomicUsize::new(0);
pub static PEAK: AtomicUsize = AtomicUsize::new(0);
pub static TOTAL_ALLOCS: AtomicUsize = AtomicUsize::new(0);
/// Cumulative bytes ever allocated (monotonic; never decremented).
pub static TOTAL_BYTES: AtomicUsize = AtomicUsize::new(0);

pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let live = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            TOTAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
            TOTAL_BYTES.fetch_add(l.size(), Ordering::Relaxed);
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // Saturating: a mismatch between the alloc and dealloc layout (which can
        // happen with Vec::from_raw_parts / shrink_to_fit) must not wrap.
        let cur = LIVE.load(Ordering::Relaxed);
        LIVE.store(cur.saturating_sub(l.size()), Ordering::Relaxed);
        System.dealloc(p, l);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = System.realloc(p, l, new);
        if !q.is_null() {
            // Subtract the old size first, then add the new one, so the running
            // total never goes negative on a shrink.
            LIVE.fetch_sub(l.size(), Ordering::Relaxed);
            let live = LIVE.fetch_add(new, Ordering::Relaxed) + new;
            TOTAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
            TOTAL_BYTES.fetch_add(new, Ordering::Relaxed);
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        q
    }
}

/// Reset the counters (call before a measured run).
pub fn reset() {
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    TOTAL_ALLOCS.store(0, Ordering::Relaxed);
    TOTAL_BYTES.store(0, Ordering::Relaxed);
}

pub fn live_mb() -> f64 {
    LIVE.load(Ordering::Relaxed) as f64 / 1e6
}

pub fn peak_mb() -> f64 {
    PEAK.load(Ordering::Relaxed) as f64 / 1e6
}

pub fn allocs() -> usize {
    TOTAL_ALLOCS.load(Ordering::Relaxed)
}

/// Cumulative bytes allocated since the last `reset`.
pub fn bytes() -> u64 {
    TOTAL_BYTES.load(Ordering::Relaxed) as u64
}
