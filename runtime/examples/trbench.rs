//! Microbenchmark the transpose kernel on the shapes the graph uses.
//!
//! These transposes are memory-bound, not index-bound: 43,840 elements is
//! 175 KB, which stays in L2, so a specialized fast path measured no better
//! than the general one. This harness is what established that.
//!
//! Run: cargo run --release --example trbench

use std::time::Instant;
use tinytts::kernels::shape;
use tinytts::onnx::Dt;
use tinytts::tensor::T;

fn bench(dims: Vec<usize>, perm: Vec<i64>, iters: usize) -> (f64, f64) {
    let n: usize = dims.iter().product();
    let v: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.01).collect();
    let x = T::from_f32(dims.clone(), v);
    // warm
    for _ in 0..3 {
        std::hint::black_box(shape::transpose(&x, &perm));
    }
    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(shape::transpose(&x, &perm));
    }
    let fast = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    // General path: force it by using a non-f32 dtype so the fast path declines.
    let xi = T {
        dt: Dt::I32,
        dims: dims.clone(),
        d: x.d.clone(),
    };
    for _ in 0..3 {
        std::hint::black_box(shape::transpose(&xi, &perm));
    }
    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(shape::transpose(&xi, &perm));
    }
    let gen = t.elapsed().as_secs_f64() / iters as f64 * 1e6;
    (fast, gen)
}

fn main() {
    let cases: Vec<(Vec<usize>, Vec<i64>)> = vec![
        (vec![1, 52, 192], vec![0, 2, 1]),
        (vec![1, 52, 4, 64], vec![0, 2, 1, 3]),
        (vec![1, 52, 4, 64], vec![0, 2, 3, 1]),
        (vec![1, 160, 274], vec![0, 2, 1]),
        (vec![1, 274, 160], vec![0, 2, 1]),
        (vec![1, 76, 1096], vec![0, 2, 1]),
        (vec![3, 2], vec![1, 0]),
        (vec![1, 192, 1], vec![0, 2, 1]),
    ];
    println!(
        "{:>22} {:>16} {:>10} {:>10} {:>8}",
        "dims", "perm", "fast us", "gen us", "speedup"
    );
    for (dims, perm) in cases {
        let n: usize = dims.iter().product();
        let iters = if n < 100_000 { 2000 } else { 200 };
        let (f, g) = bench(dims.clone(), perm.clone(), iters);
        println!(
            "{:>22} {:>16} {:>10.2} {:>10.2} {:>7.2}x",
            format!("{dims:?}"),
            format!("{perm:?}"),
            f,
            g,
            g / f
        );
    }
}
