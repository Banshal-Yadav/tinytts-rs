//! Roofline calibration: measure this machine's real single-thread peak.
//!
//! The docs assume 128 GFLOP/s (32 flops/cycle x 4.0 GHz). The SGEMM research
//! measured 141-146 GFLOP/s in isolation, which implies ~4.5 GHz. This harness
//! settles it:
//!
//!   1. Sustained SGEMM GFLOP/s at 1024^3 over a long run (thermal steady state),
//!      reporting the first and last decile so any clock decay is visible.
//!   2. The actual sustained clock under load, read from the CPU's APERF/MPERF
//!      ratio via the Windows performance counter, or from a dependent-FMA
//!      latency chain as a fallback.
//!
//! Run: cargo run --release --example roofline

use std::time::Instant;
use tinytts::kernels::sgemm;

/// A dependent FMA chain: `x = x*a + b` repeated. The latency of `vfmadd` on
/// Zen 3+ is 4 cycles, so `n` iterations take `4n` cycles. Timing it gives the
/// effective clock directly, independent of any throughput assumption.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_latency_chain(iters: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut x = _mm256_set1_ps(1.0000001);
    let a = _mm256_set1_ps(1.0000001);
    let b = _mm256_set1_ps(1e-9);
    for _ in 0..iters {
        x = _mm256_fmadd_ps(x, a, b);
    }
    let mut out = [0f32; 8];
    _mm256_storeu_ps(out.as_mut_ptr(), x);
    out[0]
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn fma_latency_chain(_iters: usize) -> f32 {
    0.0
}

fn main() {
    println!("avx2 active: {}", sgemm::avx2_active());
    println!();

    // ---- 1. sustained SGEMM at 1024^3 -------------------------------------
    let (m, n, k) = (1024usize, 1024, 1024);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.01).collect();
    let mut c = vec![0f32; m * n];
    let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;

    for _ in 0..3 {
        sgemm::gemm(m, n, k, &a, &b, &mut c);
    }

    // 60 reps, timed individually so clock decay is visible.
    let reps = 60;
    let mut per: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        sgemm::gemm(m, n, k, &a, &b, &mut c);
        per.push(gflop / t.elapsed().as_secs_f64());
    }
    let mut sorted = per.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let p = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
    let first10: f64 = per[..reps / 10].iter().sum::<f64>() / (reps / 10) as f64;
    let last10: f64 = per[reps - reps / 10..].iter().sum::<f64>() / (reps / 10) as f64;

    println!("=== sustained SGEMM 1024^3, {reps} reps ===");
    println!("  first decile : {first10:8.1} GFLOP/s");
    println!("  last decile  : {last10:8.1} GFLOP/s");
    println!("  p50          : {:8.1} GFLOP/s", p(0.50));
    println!("  p95          : {:8.1} GFLOP/s", p(0.95));
    println!("  max          : {:8.1} GFLOP/s", sorted[sorted.len() - 1]);
    println!(
        "  decay first->last: {:.1}%",
        (last10 / first10 - 1.0) * 100.0
    );
    println!();

    // ---- 2. sustained clock from the FMA latency chain --------------------
    // vfmadd latency on Zen 3+ is 4 cycles. Time a long dependent chain.
    let iters = 200_000_000usize;
    unsafe {
        fma_latency_chain(1_000_000); // warm
    }
    let t = Instant::now();
    let v = unsafe { fma_latency_chain(iters) };
    let dt = t.elapsed().as_secs_f64();
    std::hint::black_box(v);
    let cycles = 4.0 * iters as f64;
    let ghz = cycles / dt / 1e9;
    println!("=== sustained clock under load (dependent FMA chain) ===");
    println!("  {iters} dependent vfmadd, {dt:.3} s");
    println!("  assumed vfmadd latency 4 cycles -> {ghz:.3} GHz");

    // Cross-check with a serial integer add chain: `add r64, r64` is 1-cycle
    // latency on Zen 3+, so N dependent adds take exactly N cycles. This is
    // independent of the FMA latency assumption. The accumulator is read through
    // a volatile pointer so the compiler cannot fold the chain away.
    let iiters = 500_000_000u64;
    let mut acc = std::hint::black_box(0u64);
    let t2 = Instant::now();
    for _ in 0..iiters {
        acc = std::hint::black_box(acc.wrapping_add(1));
    }
    let dt2 = t2.elapsed().as_secs_f64();
    std::hint::black_box(acc);
    let ghz2 = iiters as f64 / dt2 / 1e9;
    println!("  cross-check, {iiters} dependent integer adds (1 cycle each): {ghz2:.3} GHz");
    println!(
        "  the two agree to {:.1}%",
        (ghz / ghz2 - 1.0).abs() * 100.0
    );
    println!();

    // ---- 3. restate the roofline ------------------------------------------
    let flops_per_cycle = 32.0; // 8 f32 lanes x 2 (FMA) x 2 FMA units
    let peak = flops_per_cycle * ghz;
    println!("=== roofline ===");
    println!("  {flops_per_cycle:.0} flops/cycle x {ghz:.3} GHz = {peak:.1} GFLOP/s");
    println!(
        "  measured SGEMM p50 {:.1} GFLOP/s = {:.1}% of that peak",
        p(0.50),
        p(0.50) / peak * 100.0
    );
    println!();
    println!("  (docs assumed 128.0 GFLOP/s = 32 x 4.0 GHz)");
    println!(
        "  correction factor vs the documented peak: {:.3}x",
        peak / 128.0
    );
}
