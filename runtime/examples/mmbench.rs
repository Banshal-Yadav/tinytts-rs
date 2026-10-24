//! Microbenchmark the SGEMM on the shapes the model actually uses.
//!
//! Run:  cargo run --release --example mmbench

use std::time::Instant;
use tinytts::kernels::sgemm;

fn bench(m: usize, n: usize, k: usize, iters: usize) -> f64 {
    let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.01).collect();
    let mut c = vec![0f32; m * n];
    // warmup
    for _ in 0..3 {
        sgemm::gemm(m, n, k, &a, &b, &mut c);
    }
    let t = Instant::now();
    for _ in 0..iters {
        sgemm::gemm(m, n, k, &a, &b, &mut c);
    }
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;
    gflop / dt
}

fn main() {
    println!("avx2 active: {}", sgemm::avx2_active());
    println!(
        "{:>6} {:>6} {:>6} {:>10} {:>10}",
        "m", "n", "k", "GFLOP/s", "% peak"
    );
    // The shapes the model uses, from the --shapes dump.
    let shapes = [
        (54, 256, 256),
        (54, 768, 256),
        (54, 256, 768),
        (54, 512, 512),
        (54, 128, 256),
        (54, 192, 512),
        (278, 480, 160),
        (278, 160, 480),
        (54, 64, 54),
        (54, 54, 64),
        (549, 602, 600),
        (96, 96, 96),
        (256, 256, 256),
        (512, 512, 512),
    ];
    for (m, n, k) in shapes {
        let iters = if m * n * k < 1_000_000 { 2000 } else { 200 };
        let g = bench(m, n, k, iters);
        println!("{m:>6} {n:>6} {k:>6} {g:>10.1} {:>9.1}%", g / 128.0 * 100.0);
    }
}
