// Final consolidated SGEMM benchmark.
//
//   rustc -O -C target-cpu=native scripts\sgemm_bench.rs -o scripts\sgemm_bench.exe
//   scripts\sgemm_bench.exe
//
// Implements:
//   (a) naive scalar SGEMM (ikj, the same loop order as runtime/src/kernels/matmul.rs)
//   (b) AVX2+FMA 6x16 register-blocked microkernel, packed A/B panels
//   (c) AVX2+FMA 8x8  register-blocked microkernel, packed A/B panels
//   (d) AVX2+FMA 8x8  microkernel WITHOUT packing (direct, strided B access)
// and reports achieved GFLOP/s for the shapes that dominate the model profile.
//
// Block sizes are per-kernel bests from the sweep in scripts/_blocksweep.rs.

use std::arch::x86_64::*;
use std::time::Instant;

// ---------------------------------------------------------------- naive scalar

/// ikj loop order, matching runtime/src/kernels/matmul.rs::gemm.
pub fn sgemm_naive(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for v in c[..m * n].iter_mut() {
        *v = 0.0;
    }
    for i in 0..m {
        let arow = &a[i * k..i * k + k];
        let crow = &mut c[i * n..i * n + n];
        for kk in 0..k {
            let av = arow[kk];
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

// ------------------------------------------------------------------- packing

/// A panel: MC x KC, row-major, each row padded to a multiple of 8.
/// Layout: ap[i * kc_pad + p] = A[i0 + i][k0 + p]
/// `kc_pad` MUST be `(kc + 7) & !7` for the block being packed -- not the
/// global KC. Using the global KC as the stride makes the microkernel read
/// the next row's data past the end of a short final k-block.
fn pack_a(k: usize, a: &[f32], i0: usize, k0: usize, mc: usize, kc: usize, kc_pad: usize, out: &mut [f32]) {
    for i in 0..mc {
        let src = &a[(i0 + i) * k + k0..(i0 + i) * k + k0 + kc];
        let dst = &mut out[i * kc_pad..i * kc_pad + kc];
        dst.copy_from_slice(src);
        for v in out[i * kc_pad + kc..i * kc_pad + kc_pad].iter_mut() {
            *v = 0.0;
        }
    }
}

/// B panel: KC x NC, stored as NR-wide column strips.
/// Layout: bp[p * nc_pad + j] = B[k0 + p][j0 + j]
/// This is the layout the microkernel wants: for a fixed p, the NR values
/// B[k0+p][j0..j0+NR] are contiguous.
/// `nc_pad` MUST be `(nc + 7) & !7` for the block being packed.
fn pack_b(n: usize, b: &[f32], k0: usize, j0: usize, kc: usize, nc: usize, nc_pad: usize, out: &mut [f32]) {
    for p in 0..kc {
        let src = &b[(k0 + p) * n + j0..(k0 + p) * n + j0 + nc];
        let dst = &mut out[p * nc_pad..p * nc_pad + nc];
        dst.copy_from_slice(src);
        for v in out[p * nc_pad + nc..p * nc_pad + nc_pad].iter_mut() {
            *v = 0.0;
        }
    }
}

// -------------------------------------------------------------- microkernels

/// 6x16 AVX2+FMA microkernel over packed panels.
///
/// Register budget (16 YMM on AVX2):
///   ymm0..ymm11 = 12 accumulators (6 rows x 2 vectors of 8 cols)
///   ymm12,ymm13 = the two B vectors B[k0+p][j0..j0+8] and [j0+8..j0+16]
///   ymm14       = broadcast A[i][p]
///   ymm15       = free (loop counter / prefetch)
///
/// Per k-step: 6 broadcasts + 2 B loads + 12 FMA. 12 FMA at 2 FMA/cycle = 6
/// cycles; 12 x 8 lanes x 2 flops = 192 flops / 6 cycles = 32 flops/cycle,
/// i.e. the AVX2+FMA peak. The 6 broadcasts and 2 loads fit in the 3 load
/// ports of Zen 3+, so the FMA ports stay saturated.
///
/// MEASURED (K=4096, L1-resident, single thread, this machine):
///   6x16 -> 141.7 GFLOP/s   (implied 4.43 GHz at 32 flops/cycle)
///   8x8  ->  98.8 GFLOP/s   (implied 3.09 GHz)
/// 6x16 is 1.43x faster than 8x8 in the pure-kernel measurement.
///
/// C is LOADED, accumulated into, and STORED, because the driver calls the
/// microkernel once per k-block.
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_6x16(kc: usize, ap: *const f32, kc_pad: usize, bp: *const f32, nc_pad: usize, c: *mut f32, ldc: usize) {
    let mut c00 = _mm256_loadu_ps(c.add(0 * ldc));
    let mut c01 = _mm256_loadu_ps(c.add(0 * ldc + 8));
    let mut c10 = _mm256_loadu_ps(c.add(1 * ldc));
    let mut c11 = _mm256_loadu_ps(c.add(1 * ldc + 8));
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
        let a0 = _mm256_broadcast_ss(&*ap.add(0 * kc_pad + p));
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        let a1 = _mm256_broadcast_ss(&*ap.add(1 * kc_pad + p));
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

    _mm256_storeu_ps(c.add(0 * ldc), c00);
    _mm256_storeu_ps(c.add(0 * ldc + 8), c01);
    _mm256_storeu_ps(c.add(1 * ldc), c10);
    _mm256_storeu_ps(c.add(1 * ldc + 8), c11);
    _mm256_storeu_ps(c.add(2 * ldc), c20);
    _mm256_storeu_ps(c.add(2 * ldc + 8), c21);
    _mm256_storeu_ps(c.add(3 * ldc), c30);
    _mm256_storeu_ps(c.add(3 * ldc + 8), c31);
    _mm256_storeu_ps(c.add(4 * ldc), c40);
    _mm256_storeu_ps(c.add(4 * ldc + 8), c41);
    _mm256_storeu_ps(c.add(5 * ldc), c50);
    _mm256_storeu_ps(c.add(5 * ldc + 8), c51);
}

/// 8x8 AVX2+FMA microkernel over packed panels.
///
/// Register budget (16 YMM on AVX2):
///   ymm0..ymm7  = 8 accumulators (8 rows x 8 cols)
///   ymm8        = broadcast A[i][p]
///   ymm9        = B[k0+p][j0..j0+8]
///   ymm10..15   = free (loop counters, spill, prefetch)
///
/// Per k-step: 1 broadcast + 1 load + 8 FMA. 8 x 8 lanes x 2 flops = 128
/// flops at 2 FMA/cycle = 4 cycles, i.e. 32 flops/cycle = the AVX2+FMA peak.
/// In practice it reaches only 98.8 GFLOP/s (3.09 GHz implied) because 8
/// accumulators leave too few registers for the compiler to hoist the loads
/// far enough ahead of the FMAs.
///
/// C is LOADED, accumulated into, and STORED.
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_8x8(kc: usize, ap: *const f32, kc_pad: usize, bp: *const f32, nc_pad: usize, c: *mut f32, ldc: usize) {
    let mut c0 = _mm256_loadu_ps(c.add(0 * ldc));
    let mut c1 = _mm256_loadu_ps(c.add(1 * ldc));
    let mut c2 = _mm256_loadu_ps(c.add(2 * ldc));
    let mut c3 = _mm256_loadu_ps(c.add(3 * ldc));
    let mut c4 = _mm256_loadu_ps(c.add(4 * ldc));
    let mut c5 = _mm256_loadu_ps(c.add(5 * ldc));
    let mut c6 = _mm256_loadu_ps(c.add(6 * ldc));
    let mut c7 = _mm256_loadu_ps(c.add(7 * ldc));

    let mut p = 0usize;
    while p < kc {
        let bv = _mm256_loadu_ps(bp.add(p * nc_pad));
        let a0 = _mm256_broadcast_ss(&*ap.add(0 * kc_pad + p));
        c0 = _mm256_fmadd_ps(a0, bv, c0);
        let a1 = _mm256_broadcast_ss(&*ap.add(1 * kc_pad + p));
        c1 = _mm256_fmadd_ps(a1, bv, c1);
        let a2 = _mm256_broadcast_ss(&*ap.add(2 * kc_pad + p));
        c2 = _mm256_fmadd_ps(a2, bv, c2);
        let a3 = _mm256_broadcast_ss(&*ap.add(3 * kc_pad + p));
        c3 = _mm256_fmadd_ps(a3, bv, c3);
        let a4 = _mm256_broadcast_ss(&*ap.add(4 * kc_pad + p));
        c4 = _mm256_fmadd_ps(a4, bv, c4);
        let a5 = _mm256_broadcast_ss(&*ap.add(5 * kc_pad + p));
        c5 = _mm256_fmadd_ps(a5, bv, c5);
        let a6 = _mm256_broadcast_ss(&*ap.add(6 * kc_pad + p));
        c6 = _mm256_fmadd_ps(a6, bv, c6);
        let a7 = _mm256_broadcast_ss(&*ap.add(7 * kc_pad + p));
        c7 = _mm256_fmadd_ps(a7, bv, c7);
        p += 1;
    }

    _mm256_storeu_ps(c.add(0 * ldc), c0);
    _mm256_storeu_ps(c.add(1 * ldc), c1);
    _mm256_storeu_ps(c.add(2 * ldc), c2);
    _mm256_storeu_ps(c.add(3 * ldc), c3);
    _mm256_storeu_ps(c.add(4 * ldc), c4);
    _mm256_storeu_ps(c.add(5 * ldc), c5);
    _mm256_storeu_ps(c.add(6 * ldc), c6);
    _mm256_storeu_ps(c.add(7 * ldc), c7);
}

/// 8x8 microkernel reading A and B directly from the row-major originals
/// (no packing). A is read with a broadcast from a strided row; B is read
/// with a strided load (stride = n). Used to measure the value of packing.
/// Also load-accumulate-store, for the same reason as `micro_8x8`.
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_8x8_direct(
    kc: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    let mut c0 = _mm256_loadu_ps(c.add(0 * ldc));
    let mut c1 = _mm256_loadu_ps(c.add(1 * ldc));
    let mut c2 = _mm256_loadu_ps(c.add(2 * ldc));
    let mut c3 = _mm256_loadu_ps(c.add(3 * ldc));
    let mut c4 = _mm256_loadu_ps(c.add(4 * ldc));
    let mut c5 = _mm256_loadu_ps(c.add(5 * ldc));
    let mut c6 = _mm256_loadu_ps(c.add(6 * ldc));
    let mut c7 = _mm256_loadu_ps(c.add(7 * ldc));

    let mut p = 0usize;
    while p < kc {
        let bv = _mm256_loadu_ps(b.add(p * ldb));
        let a0 = _mm256_broadcast_ss(&*a.add(0 * lda + p));
        c0 = _mm256_fmadd_ps(a0, bv, c0);
        let a1 = _mm256_broadcast_ss(&*a.add(1 * lda + p));
        c1 = _mm256_fmadd_ps(a1, bv, c1);
        let a2 = _mm256_broadcast_ss(&*a.add(2 * lda + p));
        c2 = _mm256_fmadd_ps(a2, bv, c2);
        let a3 = _mm256_broadcast_ss(&*a.add(3 * lda + p));
        c3 = _mm256_fmadd_ps(a3, bv, c3);
        let a4 = _mm256_broadcast_ss(&*a.add(4 * lda + p));
        c4 = _mm256_fmadd_ps(a4, bv, c4);
        let a5 = _mm256_broadcast_ss(&*a.add(5 * lda + p));
        c5 = _mm256_fmadd_ps(a5, bv, c5);
        let a6 = _mm256_broadcast_ss(&*a.add(6 * lda + p));
        c6 = _mm256_fmadd_ps(a6, bv, c6);
        let a7 = _mm256_broadcast_ss(&*a.add(7 * lda + p));
        c7 = _mm256_fmadd_ps(a7, bv, c7);
        p += 1;
    }

    _mm256_storeu_ps(c.add(0 * ldc), c0);
    _mm256_storeu_ps(c.add(1 * ldc), c1);
    _mm256_storeu_ps(c.add(2 * ldc), c2);
    _mm256_storeu_ps(c.add(3 * ldc), c3);
    _mm256_storeu_ps(c.add(4 * ldc), c4);
    _mm256_storeu_ps(c.add(5 * ldc), c5);
    _mm256_storeu_ps(c.add(6 * ldc), c6);
    _mm256_storeu_ps(c.add(7 * ldc), c7);
}

// ------------------------------------------------------------- blocked driver

/// Block sizes. Chosen by the sweep in scripts/_blocksweep.rs.
///
/// 6x16 kernel: KC must be <= 128 so the B panel (KC x NC x 4 B) stays in the
/// 64 KB L1d alongside the A panel row block. At KC=256 the B panel is 256 KB
/// and the kernel loses ~45% (68.5 -> 29.6 GFLOP/s on M274 N256 K966).
/// At KC=64 the packing overhead dominates (27.7 GFLOP/s).
///   MC=96, NC=192, KC=128 -> A panel 48 KB, B panel 96 KB
///
/// 8x8 kernel: much less sensitive; KC=128 is still the sweet spot.
///   MC=128, NC=256, KC=128 -> A panel 64 KB, B panel 128 KB
pub const MC_616: usize = 96;
pub const NC_616: usize = 192;
pub const KC_616: usize = 128;

pub const MC_88: usize = 128;
pub const NC_88: usize = 256;
pub const KC_88: usize = 128;

/// Packed, blocked SGEMM using the 6x16 microkernel. C is m x n row-major.
pub fn sgemm_avx2(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"), "no avx2/fma");
    let kc_pad_max = (KC_616 + 7) & !7;
    let nc_pad_max = (NC_616 + 7) & !7;
    let mut ap = vec![0f32; MC_616 * kc_pad_max];
    let mut bp = vec![0f32; kc_pad_max * nc_pad_max];

    for i0 in (0..m).step_by(MC_616) {
        let mc = MC_616.min(m - i0);
        for j0 in (0..n).step_by(NC_616) {
            let nc = NC_616.min(n - j0);
            let nc_pad = (nc + 7) & !7;
            for i in 0..mc {
                for v in c[(i0 + i) * n + j0..(i0 + i) * n + j0 + nc].iter_mut() {
                    *v = 0.0;
                }
            }
            for k0 in (0..k).step_by(KC_616) {
                let kc = KC_616.min(k - k0);
                let kc_pad = (kc + 7) & !7;
                pack_a(k, a, i0, k0, mc, kc, kc_pad, &mut ap);
                pack_b(n, b, k0, j0, kc, nc, nc_pad, &mut bp);
                unsafe {
                    let ap_ptr = ap.as_ptr();
                    let bp_ptr = bp.as_ptr();
                    let c_ptr = c.as_mut_ptr();
                    let mut i = 0usize;
                    while i + 6 <= mc {
                        let mut j = 0usize;
                        while j + 16 <= nc {
                            micro_6x16(kc, ap_ptr.add(i * kc_pad), kc_pad, bp_ptr.add(j), nc_pad, c_ptr.add((i0 + i) * n + j0 + j), n);
                            j += 16;
                        }
                        while j < nc {
                            for ii in 0..6 {
                                let mut s = 0.0f32;
                                for p in 0..kc {
                                    s += *ap_ptr.add((i + ii) * kc_pad + p) * *bp_ptr.add(p * nc_pad + j);
                                }
                                *c_ptr.add((i0 + i + ii) * n + j0 + j) += s;
                            }
                            j += 1;
                        }
                        i += 6;
                    }
                    while i < mc {
                        for j in 0..nc {
                            let mut s = 0.0f32;
                            for p in 0..kc {
                                s += *ap_ptr.add(i * kc_pad + p) * *bp_ptr.add(p * nc_pad + j);
                            }
                            *c_ptr.add((i0 + i) * n + j0 + j) += s;
                        }
                        i += 1;
                    }
                }
            }
        }
    }
}

/// Packed, blocked SGEMM using the 8x8 microkernel.
pub fn sgemm_avx2_8x8(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"), "no avx2/fma");
    let kc_pad_max = (KC_88 + 7) & !7;
    let nc_pad_max = (NC_88 + 7) & !7;
    let mut ap = vec![0f32; MC_88 * kc_pad_max];
    let mut bp = vec![0f32; kc_pad_max * nc_pad_max];

    for i0 in (0..m).step_by(MC_88) {
        let mc = MC_88.min(m - i0);
        for j0 in (0..n).step_by(NC_88) {
            let nc = NC_88.min(n - j0);
            let nc_pad = (nc + 7) & !7;
            for i in 0..mc {
                for v in c[(i0 + i) * n + j0..(i0 + i) * n + j0 + nc].iter_mut() {
                    *v = 0.0;
                }
            }
            for k0 in (0..k).step_by(KC_88) {
                let kc = KC_88.min(k - k0);
                let kc_pad = (kc + 7) & !7;
                pack_a(k, a, i0, k0, mc, kc, kc_pad, &mut ap);
                pack_b(n, b, k0, j0, kc, nc, nc_pad, &mut bp);
                unsafe {
                    let ap_ptr = ap.as_ptr();
                    let bp_ptr = bp.as_ptr();
                    let c_ptr = c.as_mut_ptr();
                    let mut i = 0usize;
                    while i + 8 <= mc {
                        let mut j = 0usize;
                        while j + 8 <= nc {
                            micro_8x8(kc, ap_ptr.add(i * kc_pad), kc_pad, bp_ptr.add(j), nc_pad, c_ptr.add((i0 + i) * n + j0 + j), n);
                            j += 8;
                        }
                        while j < nc {
                            for ii in 0..8 {
                                let mut s = 0.0f32;
                                for p in 0..kc {
                                    s += *ap_ptr.add((i + ii) * kc_pad + p) * *bp_ptr.add(p * nc_pad + j);
                                }
                                *c_ptr.add((i0 + i + ii) * n + j0 + j) += s;
                            }
                            j += 1;
                        }
                        i += 8;
                    }
                    while i < mc {
                        for j in 0..nc {
                            let mut s = 0.0f32;
                            for p in 0..kc {
                                s += *ap_ptr.add(i * kc_pad + p) * *bp_ptr.add(p * nc_pad + j);
                            }
                            *c_ptr.add((i0 + i) * n + j0 + j) += s;
                        }
                        i += 1;
                    }
                }
            }
        }
    }
}

/// Same blocking as the 8x8 packed kernel, but the microkernel reads A and B
/// directly (no packing). Measures the value of packing.
pub fn sgemm_avx2_direct(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"), "no avx2/fma");
    for v in c[..m * n].iter_mut() {
        *v = 0.0;
    }
    for i0 in (0..m).step_by(MC_88) {
        let mc = MC_88.min(m - i0);
        for j0 in (0..n).step_by(NC_88) {
            let nc = NC_88.min(n - j0);
            for k0 in (0..k).step_by(KC_88) {
                let kc = KC_88.min(k - k0);
                unsafe {
                    let a_ptr = a.as_ptr();
                    let b_ptr = b.as_ptr();
                    let c_ptr = c.as_mut_ptr();
                    let mut i = 0usize;
                    while i + 8 <= mc {
                        let mut j = 0usize;
                        while j + 8 <= nc {
                            micro_8x8_direct(kc, a_ptr.add((i0 + i) * k + k0), k, b_ptr.add(k0 * n + j0 + j), n, c_ptr.add((i0 + i) * n + j0 + j), n);
                            j += 8;
                        }
                        while j < nc {
                            for ii in 0..8 {
                                let mut s = 0.0f32;
                                for p in 0..kc {
                                    s += *a_ptr.add((i0 + i + ii) * k + k0 + p) * *b_ptr.add((k0 + p) * n + j0 + j);
                                }
                                *c_ptr.add((i0 + i + ii) * n + j0 + j) += s;
                            }
                            j += 1;
                        }
                        i += 8;
                    }
                    while i < mc {
                        for j in 0..nc {
                            let mut s = 0.0f32;
                            for p in 0..kc {
                                s += *a_ptr.add((i0 + i) * k + k0 + p) * *b_ptr.add((k0 + p) * n + j0 + j);
                            }
                            *c_ptr.add((i0 + i) * n + j0 + j) += s;
                        }
                        i += 1;
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ harness

fn fill(v: &mut [f32], seed: u64) {
    let mut s = seed | 1;
    for x in v.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = ((s >> 40) as f32 / 8388608.0) - 1.0;
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

struct Row {
    name: String,
    m: usize,
    n: usize,
    k: usize,
    naive_ms: f64,
    k616_ms: f64,
    k88_ms: f64,
    direct_ms: f64,
}

fn bench<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() / iters as f64
}

/// Pure microkernel throughput: one tile, K = 4096, panels resident in L1.
/// This is the achievable ceiling for a register-blocked kernel -- no packing,
/// no C traffic, no blocking overhead.
fn microkernel_peak(mr: usize, nr: usize, f: unsafe fn(usize, *const f32, usize, *const f32, usize, *mut f32, usize)) -> f64 {
    let kc = 4096usize;
    let kc_pad = kc;
    let nc_pad = nr;
    let ap = vec![0.5f32; mr * kc_pad];
    let bp = vec![0.25f32; kc_pad * nc_pad];
    let mut c = vec![0f32; mr * nr];
    let reps = 3000;
    unsafe { f(kc, ap.as_ptr(), kc_pad, bp.as_ptr(), nc_pad, c.as_mut_ptr(), nr) };
    let t = Instant::now();
    for _ in 0..reps {
        unsafe { f(kc, ap.as_ptr(), kc_pad, bp.as_ptr(), nc_pad, c.as_mut_ptr(), nr) };
    }
    let secs = t.elapsed().as_secs_f64() / reps as f64;
    std::hint::black_box(&c);
    2.0 * mr as f64 * nr as f64 * kc as f64 / secs / 1e9
}

fn run_shape(name: &str, m: usize, n: usize, k: usize, iters: usize) -> Row {
    let mut a = vec![0f32; m * k];
    let mut b = vec![0f32; k * n];
    fill(&mut a, 0x1234_5678);
    fill(&mut b, 0x9abc_def0);

    let mut c_ref = vec![0f32; m * n];
    let mut c_616 = vec![0f32; m * n];
    let mut c_88 = vec![0f32; m * n];
    let mut c_di = vec![0f32; m * n];

    sgemm_naive(m, n, k, &a, &b, &mut c_ref);
    sgemm_avx2(m, n, k, &a, &b, &mut c_616);
    sgemm_avx2_8x8(m, n, k, &a, &b, &mut c_88);
    sgemm_avx2_direct(m, n, k, &a, &b, &mut c_di);

    let e616 = max_abs_diff(&c_ref, &c_616);
    let e88 = max_abs_diff(&c_ref, &c_88);
    let edi = max_abs_diff(&c_ref, &c_di);
    let scale = c_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-30);
    println!(
        "  {:<28} max|C|={:>9.2}  err(6x16)={:.3e}  err(8x8)={:.3e}  err(direct)={:.3e}",
        name, scale, e616, e88, edi
    );

    let naive_ms = bench(|| sgemm_naive(m, n, k, &a, &b, &mut c_ref), iters) * 1e3;
    let k616_ms = bench(|| sgemm_avx2(m, n, k, &a, &b, &mut c_616), iters) * 1e3;
    let k88_ms = bench(|| sgemm_avx2_8x8(m, n, k, &a, &b, &mut c_88), iters) * 1e3;
    let direct_ms = bench(|| sgemm_avx2_direct(m, n, k, &a, &b, &mut c_di), iters) * 1e3;

    Row { name: name.to_string(), m, n, k, naive_ms, k616_ms, k88_ms, direct_ms }
}

fn gflops(m: usize, n: usize, k: usize, ms: f64) -> f64 {
    2.0 * (m as f64) * (n as f64) * (k as f64) / (ms * 1e-3) / 1e9
}

fn main() {
    println!("=== SGEMM microbenchmark (AVX2+FMA, single thread) ===");
    println!(
        "avx2={} fma={} avx512f={}",
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("fma"),
        is_x86_feature_detected!("avx512f")
    );
    println!("blocks 6x16: MC={} NC={} KC={}", MC_616, NC_616, KC_616);
    println!("blocks 8x8 : MC={} NC={} KC={}", MC_88, NC_88, KC_88);
    let peak616 = microkernel_peak(6, 16, micro_6x16);
    let peak88 = microkernel_peak(8, 8, micro_8x8);
    println!("pure microkernel peak (K=4096, L1-resident, single thread):");
    println!("  6x16 (12 acc): {:>6.1} GFLOP/s   implied clock @32 flops/cyc: {:.2} GHz", peak616, peak616 / 32.0);
    println!("  8x8  ( 8 acc): {:>6.1} GFLOP/s   implied clock @32 flops/cyc: {:.2} GHz", peak88, peak88 / 32.0);
    println!("  6x16 / 8x8 = {:.2}x", peak616 / peak88);
    println!();

    // Representative shapes from the model profile.
    let shapes: &[(&str, usize, usize, usize, usize)] = &[
        ("M274 N256 K966 (conv 3x3)", 274, 256, 966, 60),
        ("M137 N322 K966 (conv 3x3)", 137, 322, 966, 60),
        ("M274 N96  K288 (conv 3x3)", 274, 96, 288, 300),
        ("M137 N256 K322 (conv 1x1)", 137, 256, 322, 300),
        ("M274 N192 K576 (conv 3x3)", 274, 192, 576, 150),
        ("M137 N192 K576 (conv 3x3)", 137, 192, 576, 150),
        ("M274 N160 K1120 (dw 1x7)", 274, 160, 1120, 60),
        ("M512 N512 K512 (square ref)", 512, 512, 512, 150),
        ("M1024 N1024 K1024 (square ref)", 1024, 1024, 1024, 30),
    ];

    let mut rows = Vec::new();
    for &(name, m, n, k, iters) in shapes {
        rows.push(run_shape(name, m, n, k, iters));
    }

    println!();
    println!(
        "{:<28} {:>5} {:>5} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "shape", "M", "N", "K", "naive ms", "6x16 ms", "8x8 ms", "direct ms", "naive GF", "6x16 GF", "8x8 GF", "dir GF"
    );
    println!("{}", "-".repeat(140));
    for r in &rows {
        println!(
            "{:<28} {:>5} {:>5} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            r.name, r.m, r.n, r.k, r.naive_ms, r.k616_ms, r.k88_ms, r.direct_ms,
            gflops(r.m, r.n, r.k, r.naive_ms),
            gflops(r.m, r.n, r.k, r.k616_ms),
            gflops(r.m, r.n, r.k, r.k88_ms),
            gflops(r.m, r.n, r.k, r.direct_ms)
        );
    }

    println!();
    println!("{:<28} {:>13} {:>13} {:>13}", "shape", "speedup 6x16", "speedup 8x8", "speedup direct");
    println!("{}", "-".repeat(70));
    for r in &rows {
        println!(
            "{:<28} {:>12.1}x {:>12.1}x {:>12.1}x",
            r.name,
            r.naive_ms / r.k616_ms,
            r.naive_ms / r.k88_ms,
            r.naive_ms / r.direct_ms
        );
    }

    // Packing cost, measured in isolation.
    println!();
    let (m, n, k) = (274usize, 256usize, 966usize);
    let mut a = vec![0f32; m * k];
    let mut b = vec![0f32; k * n];
    fill(&mut a, 1);
    fill(&mut b, 2);
    let kc_pad_max = (KC_616 + 7) & !7;
    let nc_pad_max = (NC_616 + 7) & !7;
    let mut ap = vec![0f32; MC_616 * kc_pad_max];
    let mut bp = vec![0f32; kc_pad_max * nc_pad_max];
    let t = Instant::now();
    let reps = 300;
    let mut sink = 0f32;
    for _ in 0..reps {
        for i0 in (0..m).step_by(MC_616) {
            let mc = MC_616.min(m - i0);
            for k0 in (0..k).step_by(KC_616) {
                let kc = KC_616.min(k - k0);
                let kc_pad = (kc + 7) & !7;
                pack_a(k, &a, i0, k0, mc, kc, kc_pad, &mut ap);
            }
        }
        for j0 in (0..n).step_by(NC_616) {
            let nc = NC_616.min(n - j0);
            let nc_pad = (nc + 7) & !7;
            for k0 in (0..k).step_by(KC_616) {
                let kc = KC_616.min(k - k0);
                pack_b(n, &b, k0, j0, kc, nc, nc_pad, &mut bp);
            }
        }
        sink += ap[0] + bp[0];
    }
    let pack_ms = t.elapsed().as_secs_f64() / reps as f64 * 1e3;
    std::hint::black_box(sink);
    let r0 = &rows[0];
    println!("packing cost for M274 N256 K966: {:.4} ms", pack_ms);
    println!("  packed 6x16 kernel total: {:.3} ms  -> packing is {:.1}% of it", r0.k616_ms, 100.0 * pack_ms / r0.k616_ms);
    println!("  packed 8x8  kernel total: {:.3} ms  -> packing is {:.1}% of it", r0.k88_ms, 100.0 * pack_ms / r0.k88_ms);
    println!("  direct (no packing)     : {:.3} ms", r0.direct_ms);
}
