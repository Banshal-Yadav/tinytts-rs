//! SGEMM correctness: the AVX2+FMA paths vs the scalar reference.

use tinytts::kernels::sgemm;

fn naive(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f64;
            for p in 0..k {
                s += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            c[i * n + j] = s as f32;
        }
    }
    c
}

fn rng(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
}

fn check(m: usize, n: usize, k: usize) {
    let mut s = 0x1234_5678u32 ^ (m as u32) ^ ((n as u32) << 8) ^ ((k as u32) << 16);
    let a: Vec<f32> = (0..m * k).map(|_| rng(&mut s)).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng(&mut s)).collect();
    let want = naive(m, n, k, &a, &b);
    let mut got = vec![0f32; m * n];
    sgemm::gemm(m, n, k, &a, &b, &mut got);
    let mut max = 0f32;
    let mut scale = 0f32;
    for i in 0..m * n {
        max = max.max((got[i] - want[i]).abs());
        scale = scale.max(want[i].abs());
    }
    let tol = 1e-4 * scale.max(1.0);
    println!("m={m:5} n={n:5} k={k:5}  max_abs={max:.3e}  scale={scale:.2}  tol={tol:.3e}");
    assert!(max < tol, "gemm {m}x{n}x{k} err {max} >= {tol}");
}

#[test]
fn sgemm_matches_naive() {
    println!("avx2_active={}", sgemm::avx2_active());
    // model shapes
    check(274, 256, 966);
    check(137, 322, 966);
    check(274, 96, 288);
    check(137, 256, 322);
    check(274, 192, 576);
    check(137, 192, 576);
    check(274, 160, 1120);
    // edges: ragged M/N, short K, K crossing the 768 packing threshold
    check(1, 1, 1);
    check(7, 9, 13);
    check(6, 16, 128);
    check(5, 17, 767);
    check(5, 17, 768);
    check(96, 192, 128);
    check(97, 193, 129);
    check(200, 300, 1000);
}
