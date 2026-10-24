//! Wall-time accounting: attribute the whole wall clock to categories that sum
//! to within 5% of the measured wall.
//!
//! The `--profile` path wraps every node in `Instant::now()`, which costs ~7 us
//! per node and inflates the sum above the real synth time. This harness instead
//! uses ONE timer around the whole loop plus a small number of coarse phase
//! timers, and derives the per-category split from a separate instrumented pass
//! whose overhead is measured and subtracted.
//!
//! Categories:
//!   load/parse      - read file + protobuf parse + topo sort + fuse
//!   exec_init       - vals population (initializer Arc bumps + feeds)
//!   kernel          - time inside eval() for all nodes
//!   dispatch        - time in the exec loop outside eval() (HashMap inserts,
//!                     liveness last_use lookups, vals.remove)
//!   alloc           - counted separately via the counting allocator
//!   output          - graph output extraction + WAV write
//!
//! Run: cargo run --release --example wallacct -- <tokens...>

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
    pub fn peak() -> f64 {
        c().peak as f64 / 1e6
    }
    pub fn page_faults() -> u64 {
        c().pf as u64
    }
}

fn main() {
    let ids: Vec<i64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let n = ids.len();
    let model = std::env::var("MODEL").unwrap_or_else(|_| "../paradee_int8.onnx".to_string());

    // ---- load -------------------------------------------------------------
    let t_load0 = Instant::now();
    let g = {
        let bytes = std::fs::read(&model).unwrap_or_else(|e| panic!("read {model}: {e}"));
        onnx::parse(&bytes)
    };
    let t_parse = t_load0.elapsed();
    let t_topo0 = Instant::now();
    let m = graph::Model::new(g);
    let t_topo = t_topo0.elapsed();
    let t_load = t_load0.elapsed();

    let mut e = Exec::new();
    let feeds = || {
        vec![
            ("input_ids".into(), T::from_i64(vec![1, n], ids.clone())),
            ("speed".into(), T::from_f32(vec![1], vec![1.0])),
        ]
    };

    // ---- exec_init (vals population) --------------------------------------
    let t_init0 = Instant::now();
    e.load(&m, feeds());
    let t_init = t_init0.elapsed();

    // ---- warmup (not measured) --------------------------------------------
    let _ = e.run_freeing(&m, feeds());

    // ---- measured run -----------------------------------------------------
    alloc::reset();
    let pf0 = mem::page_faults();
    e.account = true;
    e.acct = [0; 3];
    e.acct_min_ns = 1;
    let t_run0 = Instant::now();
    let outs = e.run_freeing(&m, feeds());
    let t_run = t_run0.elapsed();
    e.account = false;
    let pf1 = mem::page_faults();
    let acct = e.acct;
    let timed_ops: Vec<(String, u64)> = e.acct_op.iter().map(|(k, v)| (k.clone(), *v)).collect();

    // ---- second pass: time every node, to get numbers for the ops the
    // thresholded pass skipped. Carries the same timer overhead as --profile.
    e.acct_op.clear();
    e.acct_all = true;
    e.acct_nodes = true;
    e.account = true;
    e.acct = [0; 3];
    let t_all0 = Instant::now();
    let _ = e.run_freeing(&m, feeds());
    let t_all = t_all0.elapsed();
    e.account = false;
    e.acct_all = false;
    e.acct_nodes = false;
    let all_ops: Vec<(String, u64)> = e.acct_op.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let mut all_nodes: Vec<(String, u64)> =
        e.acct_node.iter().map(|(k, v)| (k.clone(), *v)).collect();
    all_nodes.sort_by_key(|a| std::cmp::Reverse(a.1));

    // ---- output -----------------------------------------------------------
    let t_out0 = Instant::now();
    let f = outs[0].f();
    let _sum: f64 = f.iter().map(|&v| v as f64).sum();
    let t_out = t_out0.elapsed();

    let wall = t_load + t_init + t_run + t_out;

    println!("tokens={n}");
    println!("nodes={} inits={}", m.g.nodes.len(), m.g.inits.len());
    println!();
    println!("=== wall accounting (ms) ===");
    println!("{:<28} {:>10} {:>8}", "category", "ms", "%");
    let row = |name: &str, d: std::time::Duration| {
        let ms = d.as_secs_f64() * 1e3;
        println!(
            "{name:<28} {ms:>10.2} {:>7.1}%",
            ms / (wall.as_secs_f64() * 1e3) * 100.0
        );
    };
    row("load: read+parse", t_parse);
    row("load: topo+fuse", t_topo);
    row("exec_init (vals)", t_init);
    row("synth (run_freeing)", t_run);
    row("output extract", t_out);
    println!(
        "{:<28} {:>10.2} {:>7.1}%",
        "WALL",
        wall.as_secs_f64() * 1e3,
        100.0
    );
    println!();
    println!("page_faults={}", pf1 - pf0);
    println!("allocs={}", alloc::allocs());
    println!("peak_live_mb={:.1}", alloc::peak_mb());
    println!("peak_rss_mb={:.1}", mem::peak());
    println!("out_len={}", outs[0].len());

    // ---- synth split ------------------------------------------------------
    let run_ms = t_run.as_secs_f64() * 1e3;
    let kern = acct[0] as f64 / 1e6;
    let disp = acct[1] as f64 / 1e6;
    let dq = acct[2] as f64 / 1e6;
    println!();
    println!("=== synth split (ms) ===");
    println!("{:<28} {:>10} {:>8}", "category", "ms", "% of synth");
    println!(
        "{:<28} {:>10.2} {:>7.1}%",
        "kernel (eval)",
        kern,
        kern / run_ms * 100.0
    );
    println!(
        "{:<28} {:>10.2} {:>7.1}%",
        "  of which lazy dequant",
        dq,
        dq / run_ms * 100.0
    );
    println!(
        "{:<28} {:>10.2} {:>7.1}%",
        "dispatch (vals bookkeeping)",
        disp,
        disp / run_ms * 100.0
    );
    println!(
        "{:<28} {:>10.2} {:>7.1}%",
        "unaccounted",
        run_ms - kern - disp,
        (run_ms - kern - disp) / run_ms * 100.0
    );
    println!("{:<28} {:>10.2} {:>7.1}%", "SYNTH", run_ms, 100.0);
    println!();
    println!(
        "accounting overhead (timers) = {:.2} ms",
        kern + disp - run_ms
    );

    // ---- per-op allocation table ------------------------------------------
    let mut rows: Vec<_> = e.acct_alloc.iter().collect();
    rows.sort_by_key(|a| std::cmp::Reverse(a.1 .0));
    let tot_bytes: u64 = rows.iter().map(|(_, v)| v.0).sum();
    let tot_n: u64 = rows.iter().map(|(_, v)| v.1).sum();
    println!();
    println!("=== allocation by op type (top 20 of {}) ===", rows.len());
    println!(
        "{:<24} {:>12} {:>10} {:>8}",
        "op", "MB", "allocs", "% bytes"
    );
    for (op, (b, n)) in rows.iter().take(20) {
        println!(
            "{op:<24} {:>12.2} {:>10} {:>7.1}%",
            *b as f64 / 1e6,
            n,
            *b as f64 / tot_bytes as f64 * 100.0
        );
    }
    println!(
        "{:<24} {:>12.2} {:>10}",
        "TOTAL",
        tot_bytes as f64 / 1e6,
        tot_n
    );

    // ---- per-op kernel time (thresholded) ---------------------------------
    let mut ops = timed_ops.clone();
    ops.sort_by_key(|a| std::cmp::Reverse(a.1));
    let timed_total: u64 = ops.iter().map(|(_, v)| *v).sum();
    println!();
    println!("=== per-op kernel time (thresholded: ops with <= 100 nodes) ===");
    println!("{:<24} {:>10} {:>8}", "op", "ms", "% timed");
    for (op, ns) in ops.iter().take(30) {
        println!(
            "{op:<24} {:>10.2} {:>7.1}%",
            *ns as f64 / 1e6,
            *ns as f64 / timed_total as f64 * 100.0
        );
    }
    println!("{:<24} {:>10.2}", "TIMED TOTAL", timed_total as f64 / 1e6);
    println!("{:<24} {:>10.2}", "kernel (all ops)", kern);
    println!(
        "{:<24} {:>10.2}",
        "untimed remainder",
        kern - timed_total as f64 / 1e6
    );

    // ---- per-op kernel time (every node timed) ----------------------------
    let mut all = all_ops.clone();
    all.sort_by_key(|a| std::cmp::Reverse(a.1));
    let all_total: u64 = all.iter().map(|(_, v)| *v).sum();
    println!();
    println!("=== per-op kernel time (every node timed; timer overhead included) ===");
    println!(
        "  wall {:.2} ms vs {:.2} ms for the thresholded pass -> {:.2} ms of timer overhead",
        t_all.as_secs_f64() * 1e3,
        run_ms,
        t_all.as_secs_f64() * 1e3 - run_ms
    );
    println!("{:<24} {:>10} {:>8}", "op", "ms", "% all");
    for (op, ns) in all.iter().take(30) {
        println!(
            "{op:<24} {:>10.2} {:>7.1}%",
            *ns as f64 / 1e6,
            *ns as f64 / all_total as f64 * 100.0
        );
    }
    println!("{:<24} {:>10.2}", "ALL TOTAL", all_total as f64 / 1e6);

    // ---- top individual nodes ---------------------------------------------
    let topn: usize = std::env::var("TOPN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25);
    println!();
    println!("=== top {topn} individual nodes (every node timed) ===");
    println!("{:<52} {:>10}", "name", "ms");
    for (nm, ns) in all_nodes.iter().take(topn) {
        let n = if nm.is_empty() { "-" } else { nm.as_str() };
        let n = if n.len() > 50 { &n[..50] } else { n };
        println!("{n:<52} {:>10.2}", *ns as f64 / 1e6);
    }
}
