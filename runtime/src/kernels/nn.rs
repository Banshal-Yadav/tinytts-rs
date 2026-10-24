use crate::onnx::Dt;
use crate::quant::{f16_to_f32, sigmoid};
use crate::tensor::T;

pub struct LstmW {
    /// `W` transposed to `[dirs, input, 4h]`, so the gate loop reads it
    /// contiguously. ONNX stores it as `[dirs, 4h, input]`.
    pub w: Vec<f32>,
    /// `R` transposed to `[dirs, h, 4h]`. ONNX stores it as `[dirs, 4h, h]`.
    pub r: Vec<f32>,
    pub b: Vec<f32>,
    pub hidden: usize,
    pub input: usize,
    pub dirs: usize,
    pub layout: i64,
}

pub fn dequant_lstm(wq: &T, rq: &T, bq: &T, layout: i64) -> LstmW {
    let dirs = wq.dims[0];
    let hidden = wq.dims[1] / 4;
    let input = wq.dims[2];
    let g = 4 * hidden;
    // Transpose W [dirs, 4h, input] -> [dirs, input, 4h] and R [dirs, 4h, h] ->
    // [dirs, h, 4h]. The recurrence reads one input element and updates all 4h
    // gates, so the gate axis must be the contiguous one.
    let wsrc = wq.f();
    let mut w = vec![0f32; dirs * input * g];
    for d in 0..dirs {
        for j in 0..g {
            for kk in 0..input {
                w[(d * input + kk) * g + j] = wsrc[(d * g + j) * input + kk];
            }
        }
    }
    let rsrc = rq.f();
    let mut r = vec![0f32; dirs * hidden * g];
    for d in 0..dirs {
        for j in 0..g {
            for kk in 0..hidden {
                r[(d * hidden + kk) * g + j] = rsrc[(d * g + j) * hidden + kk];
            }
        }
    }
    let braw = bq.f();
    let mut b = vec![0f32; dirs * g];
    for d in 0..dirs {
        for j in 0..g {
            b[d * g + j] = braw[d * 2 * g + j] + braw[d * 2 * g + g + j];
        }
    }
    LstmW {
        w,
        r,
        b,
        hidden,
        input,
        dirs,
        layout,
    }
}

/// ONNX LSTM. Returns (Y, Y_h, Y_c).
///
/// Y   : [seq, dirs, batch, hidden]
/// Y_h : [dirs, batch, hidden] — forward: state after the last step;
///                              backward: state after consuming input t=0.
/// Y_c : same layout as Y_h, cell state.
pub fn lstm(x: &T, lw: &LstmW, seq: usize, batch: usize) -> (T, T, T) {
    let h = lw.hidden;
    let d = lw.dirs;
    let mut out = T::new(Dt::F32, vec![seq, d, batch, h]);
    let mut yh = T::new(Dt::F32, vec![d, batch, h]);
    let mut yc = T::new(Dt::F32, vec![d, batch, h]);
    let ov = out.fm();
    let yhv = yh.fm();
    let ycv = yc.fm();
    let xv = x.f();
    let mut hh = vec![0f32; batch * h];
    let mut cc = vec![0f32; batch * h];
    let mut gates = vec![0f32; batch * 4 * h];
    for di in 0..d {
        hh.iter_mut().for_each(|v| *v = 0.0);
        cc.iter_mut().for_each(|v| *v = 0.0);
        let woff = di * lw.input * 4 * h;
        let roff = di * h * 4 * h;
        let boff = di * 4 * h;
        for t in 0..seq {
            let ti = if di == 0 { t } else { seq - 1 - t };
            for bi in 0..batch {
                let xrow =
                    &xv[(ti * batch + bi) * lw.input..(ti * batch + bi) * lw.input + lw.input];
                let g = &mut gates[bi * 4 * h..(bi + 1) * 4 * h];
                g.copy_from_slice(&lw.b[boff..boff + 4 * h]);
                // W is [input, 4h] here, so each input element updates a
                // contiguous run of gates.
                for (kk, &xv_) in xrow.iter().enumerate() {
                    if xv_ == 0.0 {
                        continue;
                    }
                    let wrow = &lw.w[woff + kk * 4 * h..woff + (kk + 1) * 4 * h];
                    for (gj, &wv) in g.iter_mut().zip(wrow.iter()) {
                        *gj += xv_ * wv;
                    }
                }
                let hrow = &hh[bi * h..(bi + 1) * h];
                for (kk, &hv) in hrow.iter().enumerate() {
                    if hv == 0.0 {
                        continue;
                    }
                    let rrow = &lw.r[roff + kk * 4 * h..roff + (kk + 1) * 4 * h];
                    for (gj, &rv) in g.iter_mut().zip(rrow.iter()) {
                        *gj += hv * rv;
                    }
                }
                let c = &mut cc[bi * h..(bi + 1) * h];
                let hout = &mut hh[bi * h..(bi + 1) * h];
                for j in 0..h {
                    let i = sigmoid(g[j]);
                    let o = sigmoid(g[h + j]);
                    let f = sigmoid(g[2 * h + j]);
                    let gg = g[3 * h + j].tanh();
                    c[j] = f * c[j] + i * gg;
                    hout[j] = o * c[j].tanh();
                }
                let ob = ((ti * d + di) * batch + bi) * h;
                ov[ob..ob + h].copy_from_slice(hout);
            }
        }
        // Final state for this direction. Forward finishes at t=seq-1;
        // backward finishes at t=0, which is exactly the state left in
        // hh/cc after the loop above.
        let yb = di * batch * h;
        yhv[yb..yb + batch * h].copy_from_slice(&hh);
        ycv[yb..yb + batch * h].copy_from_slice(&cc);
    }
    (out, yh, yc)
}

pub fn layer_norm(x: &T, scale: &T, bias: &T, axis: i64, eps: f32) -> T {
    let ax = crate::tensor::norm_axis(axis, x.dims.len());
    let inner: usize = x.dims[ax..].iter().product();
    let outer = x.len() / inner;
    let mut out = T::new(Dt::F32, x.dims.clone());
    let xv = x.f();
    let ov = out.fm();
    let sv = scale.f();
    let bv = bias.f();
    for o in 0..outer {
        let row = &xv[o * inner..(o + 1) * inner];
        let mean = row.iter().sum::<f32>() / inner as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / inner as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..inner {
            ov[o * inner + i] = (row[i] - mean) * inv * sv[i] + bv[i];
        }
    }
    out
}

pub fn instance_norm(x: &T, scale: &T, bias: &T, eps: f32) -> T {
    let dims = &x.dims;
    let n = dims[0];
    let c = dims[1];
    let sz: usize = dims[2..].iter().product();
    let mut out = T::new(Dt::F32, dims.clone());
    let xv = x.f();
    let ov = out.fm();
    let sv = scale.f();
    let bv = bias.f();
    for ni in 0..n {
        for ci in 0..c {
            let base = (ni * c + ci) * sz;
            let row = &xv[base..base + sz];
            let mean = row.iter().sum::<f32>() / sz as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / sz as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..sz {
                ov[base + i] = (row[i] - mean) * inv * sv[ci] + bv[ci];
            }
        }
    }
    out
}

pub fn softmax(x: &T, axis: i64) -> T {
    let ax = crate::tensor::norm_axis(axis, x.dims.len());
    let inner: usize = x.dims[ax + 1..].iter().product();
    let axn = x.dims[ax];
    let outer = x.len() / (inner * axn);
    let mut out = T::new(Dt::F32, x.dims.clone());
    let xv = x.f();
    let ov = out.fm();
    for o in 0..outer {
        for i in 0..inner {
            let mut mx = f32::NEG_INFINITY;
            for a in 0..axn {
                mx = mx.max(xv[(o * axn + a) * inner + i]);
            }
            let mut s = 0.0;
            for a in 0..axn {
                let e = (xv[(o * axn + a) * inner + i] - mx).exp();
                ov[(o * axn + a) * inner + i] = e;
                s += e;
            }
            for a in 0..axn {
                ov[(o * axn + a) * inner + i] /= s;
            }
        }
    }
    out
}

pub fn resize_nearest(x: &T, scales: &[f32], sizes: Option<&[i64]>) -> T {
    let r = x.dims.len();
    let mut od = x.dims.clone();
    for i in 0..r {
        od[i] = match sizes {
            Some(s) => s[i] as usize,
            None => (x.dims[i] as f32 * scales[i]).floor() as usize,
        };
    }
    let mut out = T::new(Dt::F32, od.clone());
    let xv = x.f();
    let olen = out.len();
    let ov = out.fm();
    let xs = crate::tensor::strides(&x.dims);
    let os = crate::tensor::strides(&od);
    let mut idx = vec![0usize; r];
    for oi in 0..olen {
        let mut rem = oi;
        for d in 0..r {
            idx[d] = rem / os[d];
            rem %= os[d];
        }
        let mut xoff = 0usize;
        for d in 0..r {
            let src = if od[d] == x.dims[d] {
                idx[d]
            } else {
                ((idx[d] as f32 + 0.5) / (od[d] as f32 / x.dims[d] as f32)).floor() as usize
            };
            xoff += src.min(x.dims[d] - 1) * xs[d];
        }
        ov[oi] = xv[xoff];
    }
    out
}

pub fn resize_linear(x: &T, scales: &[f32], sizes: Option<&[i64]>) -> T {
    let r = x.dims.len();
    let mut od = x.dims.clone();
    for i in 0..r {
        od[i] = match sizes {
            Some(s) => s[i] as usize,
            None => (x.dims[i] as f32 * scales[i]).floor() as usize,
        };
    }
    let mut out = T::new(Dt::F32, od.clone());
    let xv = x.f();
    let olen = out.len();
    let ov = out.fm();
    let xs = crate::tensor::strides(&x.dims);
    let os = crate::tensor::strides(&od);
    let mut idx = vec![0usize; r];
    for oi in 0..olen {
        let mut rem = oi;
        for d in 0..r {
            idx[d] = rem / os[d];
            rem %= os[d];
        }
        let mut acc = 0f32;
        let mut wsum = 0f32;
        let mut corners = 1usize << r;
        for c in 0..corners {
            let mut xoff = 0usize;
            let mut w = 1f32;
            for d in 0..r {
                let sc = od[d] as f32 / x.dims[d] as f32;
                let src = (idx[d] as f32 + 0.5) / sc - 0.5;
                let lo = src.floor();
                let frac = src - lo;
                let bit = (c >> d) & 1;
                let (p, ww) = if bit == 0 {
                    (lo, 1.0 - frac)
                } else {
                    (lo + 1.0, frac)
                };
                let p = p.max(0.0).min(x.dims[d] as f32 - 1.0) as usize;
                xoff += p * xs[d];
                w *= ww;
            }
            acc += xv[xoff] * w;
            wsum += w;
        }
        ov[oi] = acc / wsum;
        corners = 0;
        let _ = corners;
    }
    out
}

pub fn f16_to_f32_slice(a: &[u16], o: &mut [f32]) {
    for (x, &y) in o.iter_mut().zip(a.iter()) {
        *x = f16_to_f32(y);
    }
}
