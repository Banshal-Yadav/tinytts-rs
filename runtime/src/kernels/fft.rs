//! Mixed-radix complex FFT (radix 2/3/5), f32, no dependencies.
//!
//! Only the sizes this model needs: 600 = 2^3 * 3 * 5^2 and 150 = 2 * 3 * 5^2.
//! Forward transform, `X[k] = sum_n x[n] * exp(-2*pi*i*k*n/N)`.
//!
//! Stockham autosort: ping-pong between two buffers, no bit-reversal pass, and
//! every twiddle is precomputed in a `Plan`. A transform allocates nothing.
//!
//! At each stage with current length `len`, stride `s` and radix `r`:
//!
//! ```text
//!   a_j = x[q + s*(p + j*m)]                       m = len/r
//!   y[q + s*(r*p + t)] = W_len^{t*p} * sum_j a_j * W_r^{j*t}
//! ```

use std::f64::consts::PI;

/// One stage of the Stockham autosort.
struct Stage {
    /// Stride between consecutive elements of a sub-transform.
    s: usize,
    /// Radix of this stage (2, 3 or 5).
    r: usize,
    /// `len / r`.
    m: usize,
    /// `tw_re[t*m + p] = Re W_len^{t*p}`, `t` in `0..r`, `p` in `0..m`.
    tw_re: Vec<f32>,
    tw_im: Vec<f32>,
}

/// A reusable FFT plan for one size.
pub struct Plan {
    n: usize,
    stages: Vec<Stage>,
    a_re: Vec<f32>,
    a_im: Vec<f32>,
    b_re: Vec<f32>,
    b_im: Vec<f32>,
    /// Half-length plan for the real-FFT trick, built once on first `rfft`.
    sub: Option<Box<Plan>>,
}

impl Plan {
    pub fn new(n: usize) -> Plan {
        assert!(is_5_smooth(n), "fft size {n} is not 5-smooth");
        let mut stages = Vec::new();
        let mut len = n;
        let mut s = 1usize;
        while len > 1 {
            let r = smallest_radix(len);
            let m = len / r;
            let mut tw_re = vec![0f32; r * m];
            let mut tw_im = vec![0f32; r * m];
            for t in 0..r {
                for p in 0..m {
                    // t*p < r*m = len, so no reduction is needed.
                    let a = -2.0 * PI * (t * p) as f64 / len as f64;
                    tw_re[t * m + p] = a.cos() as f32;
                    tw_im[t * m + p] = a.sin() as f32;
                }
            }
            stages.push(Stage {
                s,
                r,
                m,
                tw_re,
                tw_im,
            });
            len = m;
            s *= r;
        }
        Plan {
            n,
            stages,
            a_re: vec![0f32; n],
            a_im: vec![0f32; n],
            b_re: vec![0f32; n],
            b_im: vec![0f32; n],
            sub: None,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// In-place forward FFT of `re`/`im` (length `n`).
    pub fn fft(&mut self, re: &mut [f32], im: &mut [f32]) {
        let n = self.n;
        assert_eq!(re.len(), n);
        assert_eq!(im.len(), n);
        if n <= 1 {
            return;
        }
        let mut a_re = std::mem::take(&mut self.a_re);
        let mut a_im = std::mem::take(&mut self.a_im);
        let mut b_re = std::mem::take(&mut self.b_re);
        let mut b_im = std::mem::take(&mut self.b_im);
        a_re.copy_from_slice(re);
        a_im.copy_from_slice(im);

        let mut src_is_a = true;
        for st in &self.stages {
            if src_is_a {
                run_stage(st, &a_re, &a_im, &mut b_re, &mut b_im);
            } else {
                run_stage(st, &b_re, &b_im, &mut a_re, &mut a_im);
            }
            src_is_a = !src_is_a;
        }
        if src_is_a {
            re.copy_from_slice(&a_re);
            im.copy_from_slice(&a_im);
        } else {
            re.copy_from_slice(&b_re);
            im.copy_from_slice(&b_im);
        }
        self.a_re = a_re;
        self.a_im = a_im;
        self.b_re = b_re;
        self.b_im = b_im;
    }

    /// Real-input forward FFT: fills `out_re`/`out_im` with the `n/2+1` bins.
    ///
    /// Uses the real-FFT trick: pack the `n` real samples as `n/2` complex ones
    /// (`re[k] = x[2k]`, `im[k] = x[2k+1]`), run a complex FFT of length `n/2`,
    /// then unpack. That is roughly half the work of a full complex FFT of
    /// length `n`, and `n/2` is 5-smooth whenever `n` is.
    ///
    /// `X[k] = E[k] + W_n^k * O[k]` where
    /// `E[k] = (Z[k] + conj(Z[m-k]))/2`, `O[k] = (Z[k] - conj(Z[m-k]))/(2i)`,
    /// `m = n/2`, and `Z` is the length-`m` complex FFT of the packed input.
    pub fn rfft(&mut self, x: &[f32], out_re: &mut [f32], out_im: &mut [f32]) {
        let n = self.n;
        let half = n / 2 + 1;
        assert!(out_re.len() >= half && out_im.len() >= half);
        if n < 4 || !n.is_multiple_of(2) {
            // Fall back to the full complex transform for tiny or odd sizes.
            let mut re = vec![0f32; n];
            let mut im = vec![0f32; n];
            re.copy_from_slice(&x[..n]);
            self.fft(&mut re, &mut im);
            out_re[..half].copy_from_slice(&re[..half]);
            out_im[..half].copy_from_slice(&im[..half]);
            return;
        }
        let m = n / 2;
        let mut zr = vec![0f32; m];
        let mut zi = vec![0f32; m];
        for k in 0..m {
            zr[k] = x[2 * k];
            zi[k] = x[2 * k + 1];
        }
        // The half-length plan is built once and reused; building it per call
        // would cost more than the transform saves.
        if self.sub.is_none() {
            self.sub = Some(Box::new(Plan::new(m)));
        }
        let sub = self.sub.as_mut().unwrap();
        sub.fft(&mut zr, &mut zi);
        // Unpack. `Z` is periodic with period `m`, so `Z[m] = Z[0]` and
        // `Z[m-k]` for k=0 is `Z[0]`.
        for k in 0..half {
            let kk = k % m;
            let kr = (m - kk) % m;
            let (zr_k, zi_k) = (zr[kk], zi[kk]);
            let (zr_c, zi_c) = (zr[kr], -zi[kr]);
            let er = 0.5 * (zr_k + zr_c);
            let ei = 0.5 * (zi_k + zi_c);
            // O[k] = (Z[k] - conj(Z[m-k])) / (2i) = -i/2 * (dr + i*di)
            let dr = zr_k - zr_c;
            let di = zi_k - zi_c;
            let or = 0.5 * di;
            let oi = -0.5 * dr;
            // W_n^k = exp(-2*pi*i*k/n)
            let ang = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
            let (wr, wi) = (ang.cos(), ang.sin());
            out_re[k] = er + wr * or - wi * oi;
            out_im[k] = ei + wr * oi + wi * or;
        }
    }
}

fn run_stage(st: &Stage, xr: &[f32], xi: &[f32], yr: &mut [f32], yi: &mut [f32]) {
    match st.r {
        2 => stage2(st, xr, xi, yr, yi),
        3 => stage3(st, xr, xi, yr, yi),
        5 => stage5(st, xr, xi, yr, yi),
        _ => unreachable!(),
    }
}

/// Radix-2: `y[2p] = a0+a1`, `y[2p+1] = (a0-a1)*W_len^p`.
fn stage2(st: &Stage, xr: &[f32], xi: &[f32], yr: &mut [f32], yi: &mut [f32]) {
    let (s, m) = (st.s, st.m);
    for p in 0..m {
        let wr = st.tw_re[m + p];
        let wi = st.tw_im[m + p];
        let i0 = s * p;
        let i1 = s * (p + m);
        let o0 = s * 2 * p;
        for q in 0..s {
            let a0r = xr[i0 + q];
            let a0i = xi[i0 + q];
            let a1r = xr[i1 + q];
            let a1i = xi[i1 + q];
            yr[o0 + q] = a0r + a1r;
            yi[o0 + q] = a0i + a1i;
            let dr = a0r - a1r;
            let di = a0i - a1i;
            yr[o0 + s + q] = dr * wr - di * wi;
            yi[o0 + s + q] = dr * wi + di * wr;
        }
    }
}

/// Radix-3 with the standard 3-point DFT identities.
fn stage3(st: &Stage, xr: &[f32], xi: &[f32], yr: &mut [f32], yi: &mut [f32]) {
    const C3: f32 = 0.866_025_4; // sqrt(3)/2
    let (s, m) = (st.s, st.m);
    for p in 0..m {
        let (w1r, w1i) = (st.tw_re[m + p], st.tw_im[m + p]);
        let (w2r, w2i) = (st.tw_re[2 * m + p], st.tw_im[2 * m + p]);
        let i0 = s * p;
        let i1 = s * (p + m);
        let i2 = s * (p + 2 * m);
        let o0 = s * 3 * p;
        for q in 0..s {
            let a0r = xr[i0 + q];
            let a0i = xi[i0 + q];
            let a1r = xr[i1 + q];
            let a1i = xi[i1 + q];
            let a2r = xr[i2 + q];
            let a2i = xi[i2 + q];
            let s1r = a1r + a2r;
            let s1i = a1i + a2i;
            let d1r = a1r - a2r;
            let d1i = a1i - a2i;
            // t=0
            yr[o0 + q] = a0r + s1r;
            yi[o0 + q] = a0i + s1i;
            // t=1: a0 - 0.5*s1 - i*c3*d1
            let t1r = a0r - 0.5 * s1r + C3 * d1i;
            let t1i = a0i - 0.5 * s1i - C3 * d1r;
            yr[o0 + s + q] = t1r * w1r - t1i * w1i;
            yi[o0 + s + q] = t1r * w1i + t1i * w1r;
            // t=2: a0 - 0.5*s1 + i*c3*d1
            let t2r = a0r - 0.5 * s1r - C3 * d1i;
            let t2i = a0i - 0.5 * s1i + C3 * d1r;
            yr[o0 + 2 * s + q] = t2r * w2r - t2i * w2i;
            yi[o0 + 2 * s + q] = t2r * w2i + t2i * w2r;
        }
    }
}

/// Radix-5 with the standard 5-point DFT identities.
fn stage5(st: &Stage, xr: &[f32], xi: &[f32], yr: &mut [f32], yi: &mut [f32]) {
    const C1: f32 = 0.309_016_99; // cos(2pi/5)
    const S1: f32 = 0.951_056_5; // sin(2pi/5)
    const C2: f32 = -0.809_016_99; // cos(4pi/5)
    const S2: f32 = 0.587_785_25; // sin(4pi/5)
    let (s, m) = (st.s, st.m);
    for p in 0..m {
        let mut w = [(0f32, 0f32); 5];
        for t in 1..5 {
            w[t] = (st.tw_re[t * m + p], st.tw_im[t * m + p]);
        }
        let i0 = s * p;
        let i1 = s * (p + m);
        let i2 = s * (p + 2 * m);
        let i3 = s * (p + 3 * m);
        let i4 = s * (p + 4 * m);
        let o0 = s * 5 * p;
        for q in 0..s {
            let a0r = xr[i0 + q];
            let a0i = xi[i0 + q];
            let a1r = xr[i1 + q];
            let a1i = xi[i1 + q];
            let a2r = xr[i2 + q];
            let a2i = xi[i2 + q];
            let a3r = xr[i3 + q];
            let a3i = xi[i3 + q];
            let a4r = xr[i4 + q];
            let a4i = xi[i4 + q];
            let s14r = a1r + a4r;
            let s14i = a1i + a4i;
            let d14r = a1r - a4r;
            let d14i = a1i - a4i;
            let s23r = a2r + a3r;
            let s23i = a2i + a3i;
            let d23r = a2r - a3r;
            let d23i = a2i - a3i;
            // t=0
            yr[o0 + q] = a0r + s14r + s23r;
            yi[o0 + q] = a0i + s14i + s23i;
            // t=1: a0 + C1*s14 + C2*s23 - i*(S1*d14 + S2*d23)
            let t1r = a0r + C1 * s14r + C2 * s23r + S1 * d14i + S2 * d23i;
            let t1i = a0i + C1 * s14i + C2 * s23i - S1 * d14r - S2 * d23r;
            yr[o0 + s + q] = t1r * w[1].0 - t1i * w[1].1;
            yi[o0 + s + q] = t1r * w[1].1 + t1i * w[1].0;
            // t=2: a0 + C2*s14 + C1*s23 - i*(S2*d14 - S1*d23)
            let t2r = a0r + C2 * s14r + C1 * s23r + S2 * d14i - S1 * d23i;
            let t2i = a0i + C2 * s14i + C1 * s23i - S2 * d14r + S1 * d23r;
            yr[o0 + 2 * s + q] = t2r * w[2].0 - t2i * w[2].1;
            yi[o0 + 2 * s + q] = t2r * w[2].1 + t2i * w[2].0;
            // t=3: a0 + C2*s14 + C1*s23 + i*(S2*d14 - S1*d23)
            let t3r = a0r + C2 * s14r + C1 * s23r - S2 * d14i + S1 * d23i;
            let t3i = a0i + C2 * s14i + C1 * s23i + S2 * d14r - S1 * d23r;
            yr[o0 + 3 * s + q] = t3r * w[3].0 - t3i * w[3].1;
            yi[o0 + 3 * s + q] = t3r * w[3].1 + t3i * w[3].0;
            // t=4: a0 + C1*s14 + C2*s23 + i*(S1*d14 + S2*d23)
            let t4r = a0r + C1 * s14r + C2 * s23r - S1 * d14i - S2 * d23i;
            let t4i = a0i + C1 * s14i + C2 * s23i + S1 * d14r + S2 * d23r;
            yr[o0 + 4 * s + q] = t4r * w[4].0 - t4i * w[4].1;
            yi[o0 + 4 * s + q] = t4r * w[4].1 + t4i * w[4].0;
        }
    }
}

fn smallest_radix(n: usize) -> usize {
    for p in [2usize, 3, 5] {
        if n.is_multiple_of(p) {
            return p;
        }
    }
    panic!("fft size {n} is not 5-smooth");
}

fn is_5_smooth(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    for p in [2usize, 3, 5] {
        while n.is_multiple_of(p) {
            n /= p;
        }
    }
    n == 1
}

/// One-shot complex FFT (allocates a plan). Prefer `Plan` in hot loops.
pub fn fft(re: &mut [f32], im: &mut [f32]) {
    let mut p = Plan::new(re.len());
    p.fft(re, im);
}

/// One-shot real FFT (allocates a plan). Prefer `Plan` in hot loops.
pub fn rfft(x: &[f32], out_re: &mut [f32], out_im: &mut [f32]) {
    let mut p = Plan::new(x.len());
    p.rfft(x, out_re, out_im);
}
