//! A2: measure peak LIVE bytes (counting allocator) vs peak RSS, trace live
//! bytes after every node, and list the top tensors live at the peak.
//!
//! **One config per process.** `PeakWorkingSetSize` is a process-wide high-water
//! mark that never goes down, so running a no-freeing pass and then a freeing
//! pass in the same process makes the freeing pass report the no-freeing peak.
//! That artifact produced a bogus "187 MB of allocator churn" claim once. Pass
//! `FREE=0` or `FREE=1` to pick the config; the default is the freeing one, which
//! is what the binary does.
//!
//! Run:  cargo run --release --example livetrace -- <tokens...>

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
        g: usize,
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
            g: 0,
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ids: Vec<i64> = if args.is_empty() {
        vec![0, 16, 43, 44, 45, 16, 4, 0]
    } else {
        args.iter().map(|s| s.parse().unwrap()).collect()
    };
    let n = ids.len();

    let bytes = std::fs::read("../paradee_int8.onnx").unwrap();
    let g = onnx::parse(&bytes);
    let m = graph::Model::new(g);
    let mut e = Exec::new();
    e.load(
        &m,
        vec![
            ("input_ids".into(), T::from_i64(vec![1, n], ids.clone())),
            ("speed".into(), T::from_f32(vec![1], vec![1.0])),
        ],
    );
    println!("tokens={n}");
    println!(
        "after load: live={:.1} MB  rss={:.1} MB",
        alloc::live_mb(),
        mem::rss()
    );

    // Trace: run the graph in order, recording live bytes after each node.
    // One config per process: see the module comment.
    let free_dead = std::env::var("FREE").map(|v| v != "0").unwrap_or(true);
    {
        let mut e = Exec::new();
        e.load(
            &m,
            vec![
                ("input_ids".into(), T::from_i64(vec![1, n], ids.clone())),
                ("speed".into(), T::from_f32(vec![1], vec![1.0])),
            ],
        );
        alloc::reset();
        let mut peak_live = 0f64;
        let mut peak_at = 0usize;
        // Record each tensor's size at production time, so the peak report does
        // not depend on the tensor still being in `vals`.
        let mut sizes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut last_use: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for (step, &ni) in m.order.iter().enumerate() {
            for i in &m.g.nodes[ni].ins {
                if !i.is_empty() {
                    last_use.insert(i.as_str(), step);
                }
            }
        }
        let outputs: std::collections::HashSet<&str> =
            m.g.outputs.iter().map(|s| s.as_str()).collect();
        for (step, &ni) in m.order.iter().enumerate() {
            let node = &m.g.nodes[ni];
            let outs = e.eval_node(node);
            for (i, o) in node.outs.iter().enumerate() {
                if !o.is_empty() && i < outs.len() {
                    sizes.insert(o.clone(), outs[i].d.len());
                    e.put(o, outs[i].clone());
                }
            }
            if free_dead {
                for i in &node.ins {
                    if i.is_empty() || outputs.contains(i.as_str()) {
                        continue;
                    }
                    if last_use.get(i.as_str()) == Some(&step) {
                        e.vals.remove(i);
                    }
                }
            }
            let live = alloc::live_mb();
            if live > peak_live {
                peak_live = live;
                peak_at = step;
            }
        }
        println!(
            "\nfree_dead={free_dead}: peak LIVE = {:.1} MB at step {peak_at} ({}), allocs {}",
            peak_live,
            m.g.nodes[m.order[peak_at]].name,
            alloc::allocs()
        );
        println!("  peak RSS = {:.1} MB", mem::peak());

        // Top tensors live at the peak.
        let mut live_at_peak: Vec<(usize, String, String, usize, usize)> = Vec::new();
        for (step, &ni) in m.order.iter().enumerate() {
            if step > peak_at {
                break;
            }
            let node = &m.g.nodes[ni];
            for o in node.outs.iter().filter(|o| !o.is_empty()) {
                let lu = last_use.get(o.as_str()).copied().unwrap_or(usize::MAX);
                if lu >= peak_at {
                    let sz = sizes.get(o.as_str()).copied().unwrap_or(0);
                    live_at_peak.push((sz, node.op.clone(), o.clone(), step, lu));
                }
            }
        }
        live_at_peak.sort_by_key(|a| std::cmp::Reverse(a.0));
        println!("  top 10 live at the peak:");
        for (sz, op, name, p, lu) in live_at_peak.iter().take(10) {
            println!(
                "    {:8.3} MB  {op:18} {name:40} prod={p:<5} last={lu}",
                *sz as f64 / 1e6
            );
        }
        let tot: usize = live_at_peak.iter().map(|x| x.0).sum();
        let dq: Vec<_> = live_at_peak
            .iter()
            .filter(|x| x.1 == "DequantizeLinear")
            .collect();
        let dq_bytes: usize = dq.iter().map(|x| x.0).sum();
        println!(
            "  total live at peak: {:.1} MB across {} tensors; DequantizeLinear {} tensors {:.1} MB",
            tot as f64 / 1e6,
            live_at_peak.len(),
            dq.len(),
            dq_bytes as f64 / 1e6
        );
    }
}
