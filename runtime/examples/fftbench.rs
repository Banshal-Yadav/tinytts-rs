//! Microbenchmark: our FFT vs the work it does, to find the gap.

use std::time::Instant;
use tinytts::kernels::fft::Plan;

fn main() {
    let n = 600usize;
    let npos = 549usize;
    let hop = 150usize;
    let half = n / 2 + 1;

    // deterministic input
    let mut s = 0x12345678u32;
    let x: Vec<f32> = (0..npos * hop + n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    let win: Vec<f32> = (0..n)
        .map(|j| {
            let v = (std::f32::consts::PI * j as f32 / n as f32).sin();
            v * v
        })
        .collect();

    let mut plan = Plan::new(n);
    let mut seg = vec![0f32; n];
    let mut re = vec![0f32; half];
    let mut im = vec![0f32; half];
    let mut out = vec![0f32; 2 * half * npos];

    // warm
    for p in 0..npos {
        let b = p * hop;
        for j in 0..n {
            seg[j] = x[b + j] * win[j];
        }
        plan.rfft(&seg, &mut re, &mut im);
    }

    // stage 1: window multiply only
    let t0 = Instant::now();
    for _ in 0..20 {
        for p in 0..npos {
            let b = p * hop;
            for j in 0..n {
                seg[j] = x[b + j] * win[j];
            }
        }
    }
    let win_ms = t0.elapsed().as_secs_f64() * 1e3 / 20.0;

    // stage 2: rfft only
    let t0 = Instant::now();
    for _ in 0..20 {
        for _p in 0..npos {
            plan.rfft(&seg, &mut re, &mut im);
        }
    }
    let fft_ms = t0.elapsed().as_secs_f64() * 1e3 / 20.0;

    // stage 3: scatter to output layout
    let t0 = Instant::now();
    for _ in 0..20 {
        for p in 0..npos {
            for k in 0..half {
                out[k * npos + p] = re[k];
                out[(half + k) * npos + p] = im[k];
            }
        }
    }
    let scat_ms = t0.elapsed().as_secs_f64() * 1e3 / 20.0;

    // full
    let t0 = Instant::now();
    for _ in 0..20 {
        for p in 0..npos {
            let b = p * hop;
            for j in 0..n {
                seg[j] = x[b + j] * win[j];
            }
            plan.rfft(&seg, &mut re, &mut im);
            for k in 0..half {
                out[k * npos + p] = re[k];
                out[(half + k) * npos + p] = im[k];
            }
        }
    }
    let full_ms = t0.elapsed().as_secs_f64() * 1e3 / 20.0;

    println!("n={n} npos={npos}");
    println!("  window multiply : {win_ms:8.3} ms");
    println!("  rfft            : {fft_ms:8.3} ms");
    println!("  scatter         : {scat_ms:8.3} ms");
    println!("  full            : {full_ms:8.3} ms");
    println!("  (numpy full     :    3.152 ms)");
}
