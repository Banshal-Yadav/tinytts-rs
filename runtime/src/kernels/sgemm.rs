//! AVX2+FMA SGEMM with runtime feature detection and a scalar fallback.
//!
//! Two kernels, chosen by K (measured on Zen 3+):
//!
//! * K >= 768: packed 6x16 microkernel, MC=96 NC=192 KC=128
//! * K <  768: unpacked 8x8 microkernel, MC=128 NC=256 KC=128
//!
//! Both accumulate into C, so the driver zeroes (or bias-fills) C first.
//!
//! The scalar path is the reference; `tests/sgemm.rs` checks the vector paths
//! against it on randomized shapes.

use std::sync::OnceLock;

type GemmFn = unsafe fn(usize, usize, usize, *const f32, *const f32, *mut f32);

static GEMM: OnceLock<GemmFn> = OnceLock::new();

/// `C[m,n] = A[m,k] * B[k,n]`, row-major, C is overwritten (not accumulated).
pub fn gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    if m == 0 || n == 0 || k == 0 {
        c[..m * n].fill(0.0);
        return;
    }
    let f = *GEMM.get_or_init(|| {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                gemm_avx2
            } else {
                gemm_scalar
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            gemm_scalar
        }
    });
    unsafe { f(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr()) }
}

/// True when the AVX2+FMA path is active (for tests and reporting).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn avx2_active() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// The scalar path is always active on non-x86 targets.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn avx2_active() -> bool {
    false
}

// ---------------------------------------------------------------------------
// Scalar reference
// ---------------------------------------------------------------------------

unsafe fn gemm_scalar(m: usize, n: usize, k: usize, a: *const f32, b: *const f32, c: *mut f32) {
    let a = std::slice::from_raw_parts(a, m * k);
    let b = std::slice::from_raw_parts(b, k * n);
    let c = std::slice::from_raw_parts_mut(c, m * n);
    c.fill(0.0);
    for i in 0..m {
        let arow = &a[i * k..i * k + k];
        let crow = &mut c[i * n..i * n + n];
        for (kk, &av) in arow.iter().enumerate() {
            if av == 0.0 {
                continue;
            }
            let brow = &b[kk * n..kk * n + n];
            for j in 0..n {
                crow[j] += av * brow[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 + FMA
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// 6x16 microkernel over one k-block. C is loaded and accumulated.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn micro_6x16(
        kc: usize,
        ap: *const f32,
        kc_pad: usize,
        bp: *const f32,
        nc_pad: usize,
        c: *mut f32,
        ldc: usize,
    ) {
        let mut c00 = _mm256_loadu_ps(c);
        let mut c01 = _mm256_loadu_ps(c.add(8));
        let mut c10 = _mm256_loadu_ps(c.add(ldc));
        let mut c11 = _mm256_loadu_ps(c.add(ldc + 8));
        let mut c20 = _mm256_loadu_ps(c.add(2 * ldc));
        let mut c21 = _mm256_loadu_ps(c.add(2 * ldc + 8));
        let mut c30 = _mm256_loadu_ps(c.add(3 * ldc));
        let mut c31 = _mm256_loadu_ps(c.add(3 * ldc + 8));
        let mut c40 = _mm256_loadu_ps(c.add(4 * ldc));
        let mut c41 = _mm256_loadu_ps(c.add(4 * ldc + 8));
        let mut c50 = _mm256_loadu_ps(c.add(5 * ldc));
        let mut c51 = _mm256_loadu_ps(c.add(5 * ldc + 8));

        let mut p = 0usize;
        while p < kc {
            let b0 = _mm256_loadu_ps(bp.add(p * nc_pad));
            let b1 = _mm256_loadu_ps(bp.add(p * nc_pad + 8));
            let a0 = _mm256_broadcast_ss(&*ap.add(p));
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);
            let a1 = _mm256_broadcast_ss(&*ap.add(kc_pad + p));
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);
            let a2 = _mm256_broadcast_ss(&*ap.add(2 * kc_pad + p));
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);
            let a3 = _mm256_broadcast_ss(&*ap.add(3 * kc_pad + p));
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);
            let a4 = _mm256_broadcast_ss(&*ap.add(4 * kc_pad + p));
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            c41 = _mm256_fmadd_ps(a4, b1, c41);
            let a5 = _mm256_broadcast_ss(&*ap.add(5 * kc_pad + p));
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            c51 = _mm256_fmadd_ps(a5, b1, c51);
            p += 1;
        }

        _mm256_storeu_ps(c, c00);
        _mm256_storeu_ps(c.add(8), c01);
        _mm256_storeu_ps(c.add(ldc), c10);
        _mm256_storeu_ps(c.add(ldc + 8), c11);
        _mm256_storeu_ps(c.add(2 * ldc), c20);
        _mm256_storeu_ps(c.add(2 * ldc + 8), c21);
        _mm256_storeu_ps(c.add(3 * ldc), c30);
        _mm256_storeu_ps(c.add(3 * ldc + 8), c31);
        _mm256_storeu_ps(c.add(4 * ldc), c40);
        _mm256_storeu_ps(c.add(4 * ldc + 8), c41);
        _mm256_storeu_ps(c.add(5 * ldc), c50);
        _mm256_storeu_ps(c.add(5 * ldc + 8), c51);
    }

    /// 8x8 microkernel over one k-block, reading A and B directly (no packing).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn micro_8x8(
        kc: usize,
        ap: *const f32,
        lda: usize,
        bp: *const f32,
        ldb: usize,
        c: *mut f32,
        ldc: usize,
    ) {
        let mut c0 = _mm256_loadu_ps(c);
        let mut c1 = _mm256_loadu_ps(c.add(ldc));
        let mut c2 = _mm256_loadu_ps(c.add(2 * ldc));
        let mut c3 = _mm256_loadu_ps(c.add(3 * ldc));
        let mut c4 = _mm256_loadu_ps(c.add(4 * ldc));
        let mut c5 = _mm256_loadu_ps(c.add(5 * ldc));
        let mut c6 = _mm256_loadu_ps(c.add(6 * ldc));
        let mut c7 = _mm256_loadu_ps(c.add(7 * ldc));

        let mut p = 0usize;
        while p < kc {
            let b0 = _mm256_loadu_ps(bp.add(p * ldb));
            let a0 = _mm256_broadcast_ss(&*ap.add(p));
            c0 = _mm256_fmadd_ps(a0, b0, c0);
            let a1 = _mm256_broadcast_ss(&*ap.add(lda + p));
            c1 = _mm256_fmadd_ps(a1, b0, c1);
            let a2 = _mm256_broadcast_ss(&*ap.add(2 * lda + p));
            c2 = _mm256_fmadd_ps(a2, b0, c2);
            let a3 = _mm256_broadcast_ss(&*ap.add(3 * lda + p));
            c3 = _mm256_fmadd_ps(a3, b0, c3);
            let a4 = _mm256_broadcast_ss(&*ap.add(4 * lda + p));
            c4 = _mm256_fmadd_ps(a4, b0, c4);
            let a5 = _mm256_broadcast_ss(&*ap.add(5 * lda + p));
            c5 = _mm256_fmadd_ps(a5, b0, c5);
            let a6 = _mm256_broadcast_ss(&*ap.add(6 * lda + p));
            c6 = _mm256_fmadd_ps(a6, b0, c6);
            let a7 = _mm256_broadcast_ss(&*ap.add(7 * lda + p));
            c7 = _mm256_fmadd_ps(a7, b0, c7);
            p += 1;
        }

        _mm256_storeu_ps(c, c0);
        _mm256_storeu_ps(c.add(ldc), c1);
        _mm256_storeu_ps(c.add(2 * ldc), c2);
        _mm256_storeu_ps(c.add(3 * ldc), c3);
        _mm256_storeu_ps(c.add(4 * ldc), c4);
        _mm256_storeu_ps(c.add(5 * ldc), c5);
        _mm256_storeu_ps(c.add(6 * ldc), c6);
        _mm256_storeu_ps(c.add(7 * ldc), c7);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_avx2(m: usize, n: usize, k: usize, a: *const f32, b: *const f32, c: *mut f32) {
    use avx2::{micro_6x16, micro_8x8};

    if k >= 256 {
        // Packed 6x16. The 6-row blocking divides M=54 exactly (9 blocks), so
        // there is no scalar tail; the unpacked 8x8 path leaves 6 rows in a
        // scalar tail that costs more than the packing.
        const MC: usize = 96;
        const NC: usize = 192;
        const KC: usize = 128;
        let mut apack = vec![0f32; MC * KC];
        let mut bpack = vec![0f32; KC * NC];
        for i0 in (0..m).step_by(MC) {
            let im = MC.min(m - i0);
            for j0 in (0..n).step_by(NC) {
                let jn = NC.min(n - j0);
                // C tile: zero it, the microkernel accumulates.
                for i in 0..im {
                    std::ptr::write_bytes(c.add((i0 + i) * n + j0), 0, jn);
                }
                for k0 in (0..k).step_by(KC) {
                    let kn = KC.min(k - k0);
                    let kc_pad = (kn + 7) & !7;
                    let nc_pad = (jn + 15) & !15;
                    for i in 0..im {
                        let src = a.add((i0 + i) * k + k0);
                        let dst = apack.as_mut_ptr().add(i * kc_pad);
                        std::ptr::copy_nonoverlapping(src, dst, kn);
                    }
                    for p in 0..kn {
                        let src = b.add((k0 + p) * n + j0);
                        let dst = bpack.as_mut_ptr().add(p * nc_pad);
                        std::ptr::copy_nonoverlapping(src, dst, jn);
                    }
                    let mut i = 0;
                    while i + 6 <= im {
                        let mut j = 0;
                        while j + 16 <= jn {
                            micro_6x16(
                                kn,
                                apack.as_ptr().add(i * kc_pad),
                                kc_pad,
                                bpack.as_ptr().add(j),
                                nc_pad,
                                c.add((i0 + i) * n + j0 + j),
                                n,
                            );
                            j += 16;
                        }
                        while j < jn {
                            // ragged N edge: scalar
                            for r in 0..6 {
                                let mut s = 0f32;
                                for p in 0..kn {
                                    s += *apack.get_unchecked((i + r) * kc_pad + p)
                                        * *bpack.get_unchecked(p * nc_pad + j);
                                }
                                *c.add((i0 + i + r) * n + j0 + j) += s;
                            }
                            j += 1;
                        }
                        i += 6;
                    }
                    while i < im {
                        for j in 0..jn {
                            let mut s = 0f32;
                            for p in 0..kn {
                                s += *apack.get_unchecked(i * kc_pad + p)
                                    * *bpack.get_unchecked(p * nc_pad + j);
                            }
                            *c.add((i0 + i) * n + j0 + j) += s;
                        }
                        i += 1;
                    }
                }
            }
        }
    } else {
        // Unpacked 8x8.
        const MC: usize = 128;
        const NC: usize = 256;
        const KC: usize = 128;
        for i0 in (0..m).step_by(MC) {
            let im = MC.min(m - i0);
            for j0 in (0..n).step_by(NC) {
                let jn = NC.min(n - j0);
                for i in 0..im {
                    std::ptr::write_bytes(c.add((i0 + i) * n + j0), 0, jn);
                }
                for k0 in (0..k).step_by(KC) {
                    let kn = KC.min(k - k0);
                    let mut i = 0;
                    while i + 8 <= im {
                        let mut j = 0;
                        while j + 8 <= jn && j0 + j + 8 <= n {
                            micro_8x8(
                                kn,
                                a.add((i0 + i) * k + k0),
                                k,
                                b.add(k0 * n + j0 + j),
                                n,
                                c.add((i0 + i) * n + j0 + j),
                                n,
                            );
                            j += 8;
                        }
                        while j < jn {
                            for r in 0..8 {
                                let mut s = 0f32;
                                for p in 0..kn {
                                    s += *a.add((i0 + i + r) * k + k0 + p)
                                        * *b.add((k0 + p) * n + j0 + j);
                                }
                                *c.add((i0 + i + r) * n + j0 + j) += s;
                            }
                            j += 1;
                        }
                        i += 8;
                    }
                    while i < im {
                        for j in 0..jn {
                            let mut s = 0f32;
                            for p in 0..kn {
                                s += *a.add((i0 + i) * k + k0 + p) * *b.add((k0 + p) * n + j0 + j);
                            }
                            *c.add((i0 + i) * n + j0 + j) += s;
                        }
                        i += 1;
                    }
                }
            }
        }
    }
}
