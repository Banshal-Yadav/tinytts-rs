use crate::onnx::Dt;
use crate::tensor::T;

/// True when `w` is a windowed-DFT basis of the shape the `/head` vocoder uses:
/// `[2*(N/2+1), 1, N]` with `strides=[N/4]`, `pads=[0,0]`, `group=1`.
///
/// The basis is `win[n] = sin^2(pi*n/N)` times `cos`/`-sin` of `2*pi*k*n/N`,
/// blocked (all cosines, then all negated sines). Verified against the model's
/// generated weights to ~1e-9.
pub fn is_stft_basis(w: &T, strides: &[i64], pads: &[i64], groups: i64) -> Option<usize> {
    if groups != 1 || w.dims.len() != 3 || w.dims[1] != 1 {
        return None;
    }
    let n = w.dims[2];
    if w.dims[0] != n + 2 || n < 4 || !n.is_multiple_of(2) {
        return None;
    }
    let hop = strides.first().copied().unwrap_or(1);
    if hop != (n / 4) as i64 && hop != (n / 2) as i64 {
        return None;
    }
    if pads.iter().any(|&p| p != 0) {
        return None;
    }
    Some(n)
}

/// Windowed DFT of length `n`, hop `n/4`, `n/2+1` bins, over a 1-D signal.
///
/// `out[oc, p] = sum_{j<n} x[hop*p + j] * win[j] * {cos,-sin}(2*pi*k*j/n)`
/// with `oc = k` for the cosine block and `oc = n/2+1+k` for the negated-sine
/// block. Matches the model's `Conv` nodes to ~3e-5.
pub fn stft_conv(x: &T, n: usize, hop: usize) -> T {
    let half = n / 2 + 1;
    let l = x.dims[x.dims.len() - 1];
    let npos = if l >= n { (l - n) / hop + 1 } else { 0 };
    let mut out = T::new(Dt::F32, vec![1, 2 * half, npos]);
    let xv = x.f();
    let ov = out.fm();

    let win: Vec<f32> = (0..n)
        .map(|j| {
            let s = (std::f32::consts::PI * j as f32 / n as f32).sin();
            s * s
        })
        .collect();

    let mut seg = vec![0f32; n];
    let mut re = vec![0f32; half];
    let mut im = vec![0f32; half];
    let mut plan = crate::kernels::fft::Plan::new(n);
    for p in 0..npos {
        let base = p * hop;
        for j in 0..n {
            seg[j] = xv[base + j] * win[j];
        }
        plan.rfft(&seg, &mut re, &mut im);
        for k in 0..half {
            ov[k * npos + p] = re[k];
            ov[(half + k) * npos + p] = im[k];
        }
    }
    out
}

pub fn conv(
    x: &T,
    w: &T,
    b: Option<&T>,
    strides: &[i64],
    pads: &[i64],
    dil: &[i64],
    groups: i64,
) -> T {
    if b.is_none() && dil.iter().all(|&d| d == 1) {
        if let Some(n) = is_stft_basis(w, strides, pads, groups) {
            return stft_conv(x, n, strides[0] as usize);
        }
    }
    if let Some(r) = conv_im2col(x, w, b, strides, pads, dil, groups) {
        return r;
    }
    conv_naive(x, w, b, strides, pads, dil, groups)
}

/// 1-D conv with `group=1`, no dilation.
///
/// Reduces to `C[oc, p] = sum_{ic,k} W[oc, ic, k] * X[ic, p*stride - pad + k]`, i.e.
/// a GEMM with `M = Lout`, `N = Cout`, `K = Cin*Kw` and an im2col A panel (padded
/// positions are zero-filled). Returns `None` for shapes this does not cover, so
/// the caller falls back to the naive kernel.
fn conv_im2col(
    x: &T,
    w: &T,
    b: Option<&T>,
    strides: &[i64],
    pads: &[i64],
    dil: &[i64],
    groups: i64,
) -> Option<T> {
    if groups != 1 || x.dims.len() != 3 || w.dims.len() != 3 {
        return None;
    }
    if dil.iter().any(|&d| d != 1) {
        return None;
    }
    let cin = x.dims[1];
    let lin = x.dims[2];
    let cout = w.dims[0];
    let kw = w.dims[2];
    if w.dims[1] != cin {
        return None;
    }
    let stride = strides.first().copied().unwrap_or(1) as usize;
    let p0 = pads.first().copied().unwrap_or(0);
    let p1 = pads.get(1).copied().unwrap_or(p0);
    if stride == 0 {
        return None;
    }
    let eff = kw as i64;
    let lout_i = (lin as i64 + p0 + p1 - eff) / stride as i64 + 1;
    if lout_i <= 0 {
        return None;
    }
    let lout = lout_i as usize;
    let k = cin * kw;
    let xv = x.f();
    let wv = w.f();

    // im2col: A[p, ic*kw + kk] = X[ic, p*stride - p0 + kk], zero outside the input.
    let mut a = vec![0f32; lout * k];
    for ic in 0..cin {
        let xrow = &xv[ic * lin..ic * lin + lin];
        for kk in 0..kw {
            let col = ic * kw + kk;
            for p in 0..lout {
                let pos = p as i64 * stride as i64 - p0 + kk as i64;
                if pos >= 0 && (pos as usize) < lin {
                    a[p * k + col] = xrow[pos as usize];
                }
            }
        }
    }

    // B[ic*kw + kk, oc] = W[oc, ic, kk]
    let mut bm = vec![0f32; k * cout];
    for oc in 0..cout {
        let wrow = &wv[oc * k..oc * k + k];
        for (p, &wv_p) in wrow.iter().enumerate() {
            bm[p * cout + oc] = wv_p;
        }
    }

    let mut c = vec![0f32; lout * cout];
    crate::kernels::sgemm::gemm(lout, cout, k, &a, &bm, &mut c);

    let mut out = T::new(Dt::F32, vec![1, cout, lout]);
    let ov = out.fm();
    if let Some(bb) = b {
        let bv = bb.f();
        for p in 0..lout {
            for oc in 0..cout {
                ov[oc * lout + p] = c[p * cout + oc] + bv[oc];
            }
        }
    } else {
        for p in 0..lout {
            for oc in 0..cout {
                ov[oc * lout + p] = c[p * cout + oc];
            }
        }
    }
    Some(out)
}

/// Reference implementation. Kept as the oracle for the randomized differential
/// test in `tests/conv.rs`.
pub fn conv_naive(
    x: &T,
    w: &T,
    b: Option<&T>,
    strides: &[i64],
    pads: &[i64],
    dil: &[i64],
    groups: i64,
) -> T {
    let xd = &x.dims;
    let wd = &w.dims;
    let sp = xd.len();
    let g = groups as usize;
    let n = xd[0];
    let cin = if sp > 1 { xd[1] } else { 1 };
    let cout = wd[0];
    let kd: Vec<usize> = wd[2..].to_vec();
    let nd = sp - 2;
    let mut od = vec![n, cout];
    for i in 0..nd {
        let d = dil.get(i).copied().unwrap_or(1) as usize;
        let s = strides.get(i).copied().unwrap_or(1) as usize;
        let p0 = pads.get(i).copied().unwrap_or(0) as usize;
        let p1 = pads.get(i + nd).copied().unwrap_or(p0 as i64) as usize;
        let eff = d * (kd[i] - 1) + 1;
        od.push((xd[2 + i] + p0 + p1 - eff) / s + 1);
    }
    let mut out = T::new(Dt::F32, od.clone());
    let xv = x.f();
    let wv = w.f();
    let ov = out.fm();
    let cg = cin / g;
    let og = cout / g;
    let osz: usize = od[2..].iter().product();
    let xsz: usize = xd[2..].iter().product();
    let ksz: usize = kd.iter().product();
    let xs = crate::tensor::strides(&xd[2..]);
    let os = crate::tensor::strides(&od[2..]);
    let mut idx = vec![0usize; sp - 2];
    for ni in 0..n {
        for oc in 0..cout {
            let gi = oc / og;
            let obase = (ni * cout + oc) * osz;
            for oi in 0..osz {
                let mut acc = 0.0f32;
                let mut rem = oi;
                for d in 0..sp - 2 {
                    idx[d] = rem / os[d];
                    rem %= os[d];
                }
                for ic in 0..cg {
                    let xc = gi * cg + ic;
                    let xbase = (ni * cin + xc) * xsz;
                    for ki in 0..ksz {
                        let mut r = ki;
                        let mut xoff = 0i64;
                        let mut ok = true;
                        for d in (0..nd).rev() {
                            let kk = r % kd[d];
                            r /= kd[d];
                            let pos = idx[d] as i64 * strides.get(d).copied().unwrap_or(1)
                                - pads.get(d).copied().unwrap_or(0)
                                + kk as i64 * dil.get(d).copied().unwrap_or(1);
                            if pos < 0 || pos >= xd[2 + d] as i64 {
                                ok = false;
                                break;
                            }
                            xoff += pos * xs[d] as i64;
                        }
                        if !ok {
                            continue;
                        }
                        acc += xv[xbase + xoff as usize] * wv[(oc * cg + ic) * ksz + ki];
                    }
                }
                ov[obase + oi] = acc + b.map_or(0.0, |bb| bb.f()[oc]);
            }
        }
    }
    out
}

pub fn conv_transpose(
    x: &T,
    w: &T,
    b: Option<&T>,
    strides: &[i64],
    pads: &[i64],
    out_pad: &[i64],
    groups: i64,
) -> T {
    if groups == 1
        && x.dims.len() == 3
        && w.dims.len() == 3
        && w.dims[1] == 1
        && pads.iter().all(|&p| p == 0)
        && out_pad.iter().all(|&p| p == 0)
        && strides.len() == 1
    {
        return conv_transpose_gemm(x, w, b, strides[0] as usize);
    }
    conv_transpose_naive(x, w, b, strides, pads, out_pad, groups)
}

/// 1-D transposed conv with `group=1`, no padding, no output padding.
///
/// `G[i, k] = sum_ic X[ic, i] * W[ic, k]` then `out[i*stride .. i*stride+K] += G[i]`.
/// The GEMM is `[Lin, Cin] x [Cin, K]`; the overlap-add is a strided accumulate.
fn conv_transpose_gemm(x: &T, w: &T, b: Option<&T>, stride: usize) -> T {
    let cin = x.dims[1];
    let lin = x.dims[2];
    let k = w.dims[2];
    let lout = (lin - 1) * stride + k;
    let mut out = T::new(Dt::F32, vec![1, 1, lout]);
    let xv = x.f();
    let wv = w.f();
    let ov = out.fm();

    // G[i, k] = sum_ic X[ic, i] * W[ic, k]. X is [Cin, Lin] channel-major and W
    // is [Cin, K], so this is a [Lin, Cin] x [Cin, K] GEMM with A = X^T.
    // Transpose X into a contiguous [Lin, Cin] panel once, then run the SGEMM.
    let mut a = vec![0f32; lin * cin];
    for ic in 0..cin {
        let xrow = &xv[ic * lin..ic * lin + lin];
        for (i, &v) in xrow.iter().enumerate() {
            a[i * cin + ic] = v;
        }
    }
    let mut g = vec![0f32; lin * k];
    crate::kernels::sgemm::gemm(lin, k, cin, &a, wv, &mut g);

    // Overlap-add. First input position assigns, the rest accumulate, so the
    // output buffer needs no zero-fill.
    ov[..k].copy_from_slice(&g[..k]);
    for i in 1..lin {
        let base = i * stride;
        let grow = &g[i * k..i * k + k];
        for (o, &gv) in ov[base..base + k].iter_mut().zip(grow.iter()) {
            *o += gv;
        }
    }

    if let Some(bb) = b {
        let bv = bb.f();
        for (o, &bv) in ov.iter_mut().zip(bv.iter()) {
            *o += bv;
        }
    }
    out
}

/// Reference implementation. Kept as the oracle for the randomized differential
/// test in `tests/conv.rs`.
pub fn conv_transpose_naive(
    x: &T,
    w: &T,
    b: Option<&T>,
    strides: &[i64],
    pads: &[i64],
    out_pad: &[i64],
    groups: i64,
) -> T {
    let xd = &x.dims;
    let wd = &w.dims;
    let sp = xd.len();
    let g = groups as usize;
    let n = xd[0];
    let cin = xd[1];
    let cout = wd[1] * g;
    let kd: Vec<usize> = wd[2..].to_vec();
    let mut od = vec![n, cout];
    for i in 0..sp - 2 {
        let s = strides[i] as usize;
        let p0 = pads[i] as usize;
        let p1 = pads[i + sp - 2] as usize;
        let op = out_pad.get(i).copied().unwrap_or(0) as usize;
        od.push((xd[2 + i] - 1) * s + kd[i] - p0 - p1 + op);
    }
    let mut out = T::new(Dt::F32, od.clone());
    let xv = x.f();
    let wv = w.f();
    let ov = out.fm();
    let cg = cin / g;
    let og = cout / g;
    let osz: usize = od[2..].iter().product();
    let xsz: usize = xd[2..].iter().product();
    let ksz: usize = kd.iter().product();
    let xs = crate::tensor::strides(&xd[2..]);
    let os = crate::tensor::strides(&od[2..]);
    let mut idx = vec![0usize; sp - 2];
    for ni in 0..n {
        for ic in 0..cin {
            let gi = ic / cg;
            let xbase = (ni * cin + ic) * xsz;
            for ii in 0..xsz {
                let mut rem = ii;
                for d in 0..sp - 2 {
                    idx[d] = rem / xs[d];
                    rem %= xs[d];
                }
                let xval = xv[xbase + ii];
                if xval == 0.0 {
                    continue;
                }
                for oc in 0..og {
                    let ocg = gi * og + oc;
                    let obase = (ni * cout + ocg) * osz;
                    for ki in 0..ksz {
                        let mut r = ki;
                        let mut ooff = 0i64;
                        let mut ok = true;
                        for d in (0..sp - 2).rev() {
                            let kk = r % kd[d];
                            r /= kd[d];
                            let pos = idx[d] as i64 * strides[d] - pads[d] + kk as i64;
                            if pos < 0 || pos >= od[2 + d] as i64 {
                                ok = false;
                                break;
                            }
                            ooff += pos * os[d] as i64;
                        }
                        if !ok {
                            continue;
                        }
                        ov[obase + ooff as usize] += xval * wv[(ic * og + oc) * ksz + ki];
                    }
                }
            }
        }
    }
    if let Some(bb) = b {
        let bv = bb.f();
        for ni in 0..n {
            for oc in 0..cout {
                let obase = (ni * cout + oc) * osz;
                for v in &mut ov[obase..obase + osz] {
                    *v += bv[oc];
                }
            }
        }
    }
    out
}
