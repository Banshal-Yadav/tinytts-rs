//! Elementwise fast paths vs the general strided broadcast path.

// chunks_exact(N) is the idiomatic way to reinterpret a byte buffer as a
// fixed-width scalar; the suggested alternative is not clearer.
#![allow(clippy::chunks_exact_to_as_chunks)]
use tinytts::kernels::elem;
use tinytts::onnx::Dt;
use tinytts::tensor::T;

fn rng(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
}

fn mk(dims: &[usize], seed: &mut u32) -> T {
    let n: usize = dims.iter().product();
    let v: Vec<f32> = (0..n).map(|_| rng(seed)).collect();
    T::from_f32(dims.to_vec(), v)
}

/// The general path, forced by using a dtype the fast paths do not cover.
fn general(a: &T, b: &T, op: u8) -> T {
    let af = T {
        dt: Dt::F64,
        dims: a.dims.clone(),
        d: a.f()
            .iter()
            .flat_map(|v| (*v as f64).to_le_bytes())
            .collect::<Vec<u8>>()
            .into(),
    };
    let bf = T {
        dt: Dt::F64,
        dims: b.dims.clone(),
        d: b.f()
            .iter()
            .flat_map(|v| (*v as f64).to_le_bytes())
            .collect::<Vec<u8>>()
            .into(),
    };
    let r = elem::binary(&af, &bf, op);
    let vals: Vec<f32> =
        r.d.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect();
    T::from_f32(r.dims.clone(), vals)
}

fn check(ad: &[usize], bd: &[usize], op: u8) {
    let mut s = 0x9e3779b9u32 ^ (ad.len() as u32) ^ ((bd.len() as u32) << 4) ^ ((op as u32) << 8);
    let a = mk(ad, &mut s);
    let b = mk(bd, &mut s);
    let fast = elem::binary(&a, &b, op);
    let want = general(&a, &b, op);
    assert_eq!(fast.dims, want.dims, "shape {ad:?} {bd:?} op={op}");
    let fv = fast.f();
    let wv = want.f();
    let mut max = 0f32;
    for i in 0..fv.len() {
        max = max.max((fv[i] - wv[i]).abs());
    }
    println!("{ad:?} x {bd:?} op={op}  n={}  max_abs={max:.3e}", fv.len());
    assert!(max < 1e-5, "mismatch {ad:?} {bd:?} op={op}: {max}");
}

#[test]
fn binary_fast_paths_match_general() {
    for op in [0u8, 1, 2, 3, 4, 5] {
        // same shape
        check(&[1, 64, 82200], &[1, 64, 82200], op);
        check(&[3, 5], &[3, 5], op);
        // scalar
        check(&[1, 64, 82200], &[1], op);
        check(&[1], &[1, 64, 82200], op);
        // last-dim broadcast
        check(&[1, 64, 82200], &[1, 64, 1], op);
        // per-channel [1,C,1] x [1,C,L]
        check(&[1, 64, 82200], &[1, 64, 1], op);
        // outer product [1,1,L] x [1,C,1]
        check(&[1, 1, 82200], &[1, 64, 1], op);
        check(&[1, 64, 1], &[1, 1, 82200], op);
        // general (falls through)
        check(&[1, 64, 82200], &[1, 1, 82200], op);
        check(&[2, 3, 4], &[1, 4], op);
    }
}
