//! Break down the fixed RSS floor: weights, dequantized weights, FFT plans,
//! SGEMM packing, and the live activation set at a minimal input.

use tinytts::{exec::Exec, graph, onnx, tensor::T};

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
        fn K32GetProcessMemoryInfo(p: *mut c_void, c: *mut Pmc, cb: u32) -> i32;
    }
    pub fn rss_mb() -> f64 {
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
        c.working_set_size as f64 / 1e6
    }
    pub fn peak_mb() -> f64 {
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
        c.peak_working_set_size as f64 / 1e6
    }
}

fn main() {
    println!("start                       rss={:6.1} MB", mem::rss_mb());

    let g = {
        let bytes = std::fs::read("../paradee_int8.onnx").unwrap();
        println!(
            "after read model file       rss={:6.1} MB  (file {:.2} MB)",
            mem::rss_mb(),
            bytes.len() as f64 / 1e6
        );
        onnx::parse(&bytes)
    };
    println!("after parse (graph+inits)   rss={:6.1} MB", mem::rss_mb());

    // Sum the initializer bytes as stored (int8/fp16) and as f32.
    let mut stored = 0usize;
    let mut as_f32 = 0usize;
    for (_, t) in &g.inits {
        stored += t.data.len();
        as_f32 += t.dims.iter().product::<usize>() * 4;
    }
    println!("  initializers stored       {:6.2} MB", stored as f64 / 1e6);
    println!("  initializers as f32       {:6.2} MB", as_f32 as f64 / 1e6);

    let m = graph::Model::new(g);
    println!("after Model::new (topo)     rss={:6.1} MB", mem::rss_mb());

    let mut e = Exec::new();
    e.load(
        &m,
        vec![
            ("input_ids".into(), T::from_i64(vec![1, 2], vec![0, 4])),
            ("speed".into(), T::from_f32(vec![1], vec![1.0])),
        ],
    );
    println!("after Exec::load (vals)     rss={:6.1} MB", mem::rss_mb());

    // Sum what vals actually holds.
    let mut vals_bytes = 0usize;
    for t in e.vals.values() {
        vals_bytes += t.d.len();
    }
    println!(
        "  vals holds                {:6.2} MB",
        vals_bytes as f64 / 1e6
    );

    let outs = e.run_freeing(
        &m,
        vec![
            ("input_ids".into(), T::from_i64(vec![1, 2], vec![0, 4])),
            ("speed".into(), T::from_f32(vec![1], vec![1.0])),
        ],
    );
    println!(
        "after run (2 tokens)        rss={:6.1} MB  peak={:6.1} MB",
        mem::rss_mb(),
        mem::peak_mb()
    );
    println!("  output len {}", outs[0].len());
}
