//! FFT correctness: mixed-radix f32 FFT vs a naive O(n^2) DFT.

use tinytts::kernels::fft;

fn naive_dft(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    let mut re = vec![0f32; n];
    let mut im = vec![0f32; n];
    for k in 0..n {
        let mut sr = 0f64;
        let mut si = 0f64;
        for (t, &v) in x.iter().enumerate() {
            let a = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
            sr += v as f64 * a.cos();
            si += v as f64 * a.sin();
        }
        re[k] = sr as f32;
        im[k] = si as f32;
    }
    (re, im)
}

fn check(n: usize) {
    // deterministic pseudo-random input
    let mut s = 0x12345678u32;
    let x: Vec<f32> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    let (er, ei) = naive_dft(&x);
    let mut re = x.clone();
    let mut im = vec![0f32; n];
    fft::fft(&mut re, &mut im);
    let mut max = 0f32;
    for k in 0..n {
        max = max.max((re[k] - er[k]).abs()).max((im[k] - ei[k]).abs());
    }
    // f32 round-off grows with n; scale by n
    let tol = 1e-4 * n as f32;
    println!("n={n:5}  max_abs_err={max:.3e}  tol={tol:.3e}");
    assert!(max < tol, "fft n={n} err {max} >= {tol}");
}

#[test]
fn fft_matches_naive_dft() {
    for n in [
        2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 25, 30, 32, 50, 75, 100, 150, 600,
    ] {
        check(n);
    }
}

#[test]
fn rfft_matches_naive_dft() {
    let n = 600;
    let mut s = 0xdeadbeefu32;
    let x: Vec<f32> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    let (er, ei) = naive_dft(&x);
    let half = n / 2 + 1;
    let mut re = vec![0f32; half];
    let mut im = vec![0f32; half];
    fft::rfft(&x, &mut re, &mut im);
    let mut max = 0f32;
    for k in 0..half {
        max = max.max((re[k] - er[k]).abs()).max((im[k] - ei[k]).abs());
    }
    println!("rfft n={n} max_abs_err={max:.3e}");
    assert!(max < 1e-2, "rfft err {max}");
}
