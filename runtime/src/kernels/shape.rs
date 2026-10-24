use crate::onnx::Dt;
use crate::tensor::{norm_axis, strides, T};

pub fn reshape(x: &T, shape: &[i64]) -> T {
    let mut d = Vec::with_capacity(shape.len());
    let mut neg = None;
    for (i, &s) in shape.iter().enumerate() {
        if s == 0 {
            d.push(*x.dims.get(i).unwrap_or(&1));
        } else if s < 0 {
            neg = Some(i);
            d.push(1);
        } else {
            d.push(s as usize);
        }
    }
    if let Some(i) = neg {
        let known: usize = d.iter().product();
        d[i] = x.len() / known;
    }
    T {
        dt: x.dt,
        dims: d,
        d: x.d.clone(),
    }
}

pub fn transpose(x: &T, perm: &[i64]) -> T {
    let xr = x.dims.len();
    if xr <= 1 {
        return T {
            dt: x.dt,
            dims: x.dims.clone(),
            d: x.d.clone(),
        };
    }
    let r = perm.len().max(xr);
    let mut p2 = perm.to_vec();
    if perm.len() < xr {
        let used: std::collections::HashSet<i64> = perm.iter().copied().collect();
        for i in 0..xr as i64 {
            if !used.contains(&i) {
                p2.push(i);
            }
        }
    }
    let od: Vec<usize> = p2.iter().map(|&p| x.dims[norm_axis(p, xr)]).collect();
    let mut out = T::new(x.dt, od.clone());
    let xs = strides(&x.dims);
    let os = strides(&od);
    let esz = x.dt.size();
    let mut idx = vec![0usize; r];
    for oi in 0..out.len() {
        let mut rem = oi;
        for d in 0..r {
            idx[d] = rem / os[d];
            rem %= os[d];
        }
        let mut xoff = 0usize;
        for d in 0..r {
            xoff += idx[d] * xs[norm_axis(p2[d], xr)];
        }
        out.dm()[oi * esz..(oi + 1) * esz].copy_from_slice(&x.d[xoff * esz..(xoff + 1) * esz]);
    }
    out
}

pub fn slice(x: &T, starts: &[i64], ends: &[i64], axes: &[i64], steps: &[i64]) -> T {
    let r = x.dims.len();
    if r == 0 {
        return T {
            dt: x.dt,
            dims: vec![],
            d: x.d.clone(),
        };
    }
    let mut st = vec![0i64; r];
    let mut en: Vec<i64> = x.dims.iter().map(|&d| d as i64).collect();
    let mut sp = vec![1i64; r];
    for (i, &a) in axes.iter().enumerate() {
        let ax = norm_axis(a, r);
        let dim = x.dims[ax] as i64;
        let s = steps.get(i).copied().unwrap_or(1);
        let mut b = starts[i];
        let mut e = ends[i];
        if b < 0 {
            b += dim;
        }
        if e < 0 {
            e += dim;
        }
        if s > 0 {
            b = b.clamp(0, dim);
            e = e.clamp(0, dim);
        } else {
            b = b.clamp(0, dim - 1);
            e = e.clamp(-1, dim - 1);
        }
        st[ax] = b;
        en[ax] = e;
        sp[ax] = s;
    }
    let mut od = vec![0usize; r];
    for d in 0..r {
        let n = if sp[d] > 0 {
            (en[d] - st[d] + sp[d] - 1) / sp[d]
        } else if sp[d] < 0 {
            (st[d] - en[d] + (-sp[d]) - 1) / (-sp[d])
        } else {
            0
        };
        od[d] = n.max(0) as usize;
    }
    let mut out = T::new(x.dt, od.clone());
    let xs = strides(&x.dims);
    let os = strides(&od);
    let esz = x.dt.size();
    // Fast path: a single-axis slice with step 1 over a contiguous run. The
    // `/head` STFT blocks are exactly this (`[0:301]` and `[301:602]` on the row
    // axis of a `[1,602,frames]` tensor), and the general path below pays a `/`
    // and a `%` per element for them.
    if let Some(()) = slice_contiguous(x, &st, &en, &sp, &od, &xs, &os, esz, &mut out) {
        return out;
    }
    let mut idx = vec![0usize; r];
    for oi in 0..out.len() {
        let mut rem = oi;
        for d in 0..r {
            idx[d] = rem / os[d];
            rem %= os[d];
        }
        let mut xoff = 0isize;
        for d in 0..r {
            xoff += (st[d] + idx[d] as i64 * sp[d]) as isize * xs[d] as isize;
        }
        out.dm()[oi * esz..(oi + 1) * esz]
            .copy_from_slice(&x.d[xoff as usize * esz..(xoff as usize + 1) * esz]);
    }
    out
}

/// Bulk-copy a slice when it is a contiguous run along one axis.
///
/// Returns `Some(())` when it handled the copy. The condition is: exactly one
/// axis is narrowed, its step is 1, and the narrowed axis is the outermost one
/// that is not fully selected — so the selected elements form `outer` runs of
/// `inner` contiguous elements each.
#[allow(clippy::too_many_arguments)]
fn slice_contiguous(
    x: &T,
    st: &[i64],
    _en: &[i64],
    sp: &[i64],
    od: &[usize],
    xs: &[usize],
    _os: &[usize],
    esz: usize,
    out: &mut T,
) -> Option<()> {
    let r = x.dims.len();
    // Find the narrowed axes.
    let mut narrowed: Vec<usize> = Vec::new();
    for d in 0..r {
        if sp[d] != 1 {
            return None;
        }
        if od[d] != x.dims[d] {
            narrowed.push(d);
        }
    }
    if narrowed.len() != 1 {
        return None;
    }
    let ax = narrowed[0];
    // Everything before `ax` must be fully selected, so the runs are contiguous.
    for d in 0..ax {
        if od[d] != x.dims[d] {
            return None;
        }
    }
    let outer: usize = x.dims[..ax].iter().product();
    let inner: usize = x.dims[ax + 1..].iter().product();
    let take = od[ax];
    let src_run = x.dims[ax];
    let start = st[ax] as usize;
    let dst = out.dm();
    for o in 0..outer {
        let s = (o * src_run + start) * inner * esz;
        let d = o * take * inner * esz;
        dst[d..d + take * inner * esz].copy_from_slice(&x.d[s..s + take * inner * esz]);
    }
    let _ = xs;
    Some(())
}

pub fn gather(x: &T, idx: &T, axis: i64) -> T {
    if x.dims.is_empty() {
        let esz = x.dt.size();
        let mut out = T::new(x.dt, idx.dims.clone());
        for j in 0..idx.len() {
            out.dm()[j * esz..(j + 1) * esz].copy_from_slice(&x.d[..esz]);
        }
        return out;
    }
    let ax = norm_axis(axis, x.dims.len());
    let outer: usize = x.dims[..ax].iter().product();
    let inner: usize = x.dims[ax + 1..].iter().product();
    let dim = x.dims[ax];
    let nidx = idx.len();
    let mut od = x.dims[..ax].to_vec();
    od.extend_from_slice(&idx.dims);
    od.extend_from_slice(&x.dims[ax + 1..]);
    let mut out = T::new(x.dt, od);
    let esz = x.dt.size();
    let iv: Vec<i64> = match idx.dt {
        Dt::I64 => idx.i64s().to_vec(),
        Dt::I32 => idx.i32s().iter().map(|&v| v as i64).collect(),
        Dt::F32 => idx.f().iter().map(|&v| v as i64).collect(),
        _ => panic!("gather idx {:?}", idx.dt),
    };
    for o in 0..outer {
        for (j, &i) in iv.iter().enumerate() {
            let i = if i < 0 { i + dim as i64 } else { i } as usize;
            let src = (o * dim + i) * inner;
            let dst = (o * nidx + j) * inner;
            out.dm()[dst * esz..(dst + inner) * esz]
                .copy_from_slice(&x.d[src * esz..(src + inner) * esz]);
        }
    }
    out
}

pub fn concat(ts: &[&T], axis: i64) -> T {
    if ts[0].dims.is_empty() {
        let esz = ts[0].dt.size();
        let mut out = T::new(ts[0].dt, vec![ts.len()]);
        for (off, t) in ts.iter().enumerate() {
            if !t.d.is_empty() {
                out.dm()[off * esz..(off + 1) * esz].copy_from_slice(&t.d[..esz]);
            }
        }
        return out;
    }
    let ax = norm_axis(axis, ts[0].dims.len());
    let mut od = ts[0].dims.clone();
    od[ax] = ts.iter().map(|t| t.dims[ax]).sum();
    let mut out = T::new(ts[0].dt, od.clone());
    let outer: usize = od[..ax].iter().product();
    let inner: usize = od[ax + 1..].iter().product();
    let esz = ts[0].dt.size();
    let mut off = 0usize;
    for t in ts {
        let w = t.dims[ax] * inner;
        for o in 0..outer {
            let dst = (o * od[ax] + off) * inner;
            let src = o * w;
            out.dm()[dst * esz..(dst + w) * esz]
                .copy_from_slice(&t.d[(src) * esz..(src + w) * esz]);
        }
        off += t.dims[ax];
    }
    out
}

pub fn unsqueeze(x: &T, axes: &[i64]) -> T {
    let mut d = x.dims.clone();
    let mut ax: Vec<usize> = axes
        .iter()
        .map(|&a| norm_axis(a, x.dims.len() + axes.len()))
        .collect();
    ax.sort_unstable();
    for a in ax {
        d.insert(a, 1);
    }
    T {
        dt: x.dt,
        dims: d,
        d: x.d.clone(),
    }
}

pub fn squeeze(x: &T, axes: &[i64]) -> T {
    let r = x.dims.len();
    let mut d = Vec::new();
    if axes.is_empty() {
        for &v in &x.dims {
            if v != 1 {
                d.push(v);
            }
        }
    } else {
        let ax: Vec<usize> = axes.iter().map(|&a| norm_axis(a, r)).collect();
        for (i, &v) in x.dims.iter().enumerate() {
            if !ax.contains(&i) {
                d.push(v);
            }
        }
    }
    T {
        dt: x.dt,
        dims: d,
        d: x.d.clone(),
    }
}

pub fn expand(x: &T, shape: &[i64]) -> T {
    let r = shape.len();
    let mut od = vec![0usize; r];
    for i in 0..r {
        let s = shape[i];
        od[i] = if s <= 0 {
            if r >= x.dims.len() && i >= r - x.dims.len() {
                x.dims[i - (r - x.dims.len())]
            } else {
                1
            }
        } else {
            s as usize
        };
    }
    let mut xd = vec![1usize; r];
    if x.dims.len() <= r {
        let off = r - x.dims.len();
        xd[off..].copy_from_slice(&x.dims);
    } else {
        for i in 0..r {
            xd[i] = x.dims[x.dims.len() - r + i];
        }
    }
    for i in 0..r {
        if od[i] == 1 && xd[i] != 1 {
            od[i] = xd[i];
        }
    }
    let mut out = T::new(x.dt, od.clone());
    let os = strides(&od);
    let xs = strides(&xd);
    let esz = x.dt.size();
    let mut idx = vec![0usize; r];
    for oi in 0..out.len() {
        let mut rem = oi;
        for d in 0..r {
            idx[d] = rem / os[d];
            rem %= os[d];
        }
        let mut xoff = 0usize;
        for d in 0..r {
            let p = if xd[d] == 1 { 0 } else { idx[d] };
            xoff += p * xs[d];
        }
        out.dm()[oi * esz..(oi + 1) * esz].copy_from_slice(&x.d[xoff * esz..(xoff + 1) * esz]);
    }
    out
}

/// ONNX Pad. `mode` is 0=constant, 1=reflect, 2=edge.
/// Never silently falls back: an unknown mode panics.
pub fn pad(x: &T, pads: &[i64], val: f32, mode: u8) -> T {
    let r = x.dims.len();
    let mut od = vec![0usize; r];
    for i in 0..r {
        od[i] = (x.dims[i] as i64 + pads[i] + pads[i + r]) as usize;
    }
    let mut out = T::new(x.dt, od.clone());
    let esz = x.dt.size();
    let xs = strides(&x.dims);
    let os = strides(&od);
    let mut idx = vec![0usize; r];

    // Constant mode: fill with `val` (dtype-aware), then copy the interior.
    if mode == 0 {
        match x.dt {
            Dt::F32 => out.fm().fill(val),
            Dt::F64 => {
                let o = unsafe {
                    std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut f64, out.len())
                };
                o.fill(val as f64);
            }
            Dt::I64 => {
                let o = unsafe {
                    std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i64, out.len())
                };
                o.fill(val as i64);
            }
            Dt::I32 => {
                let o = unsafe {
                    std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i32, out.len())
                };
                o.fill(val as i32);
            }
            Dt::F16 => {
                let o = unsafe {
                    std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut u16, out.len())
                };
                o.fill(crate::quant::f32_to_f16(val));
            }
            Dt::I8 => out.dm().fill(val as i8 as u8),
            Dt::U8 | Dt::Bool => out.dm().fill(val as u8),
        }
    }

    for oi in 0..out.len() {
        let mut rem = oi;
        for d in 0..r {
            idx[d] = rem / os[d];
            rem %= os[d];
        }
        // Map each output index back to an input index per the pad mode.
        let mut src = 0usize;
        let mut inside = true;
        for d in 0..r {
            let dim = x.dims[d] as i64;
            let mut p = idx[d] as i64 - pads[d];
            if p < 0 || p >= dim {
                match mode {
                    0 => {
                        inside = false;
                        break;
                    }
                    1 => {
                        // reflect: mirror without repeating the edge sample
                        if dim == 1 {
                            p = 0;
                        } else {
                            let period = 2 * (dim - 1);
                            p = p.rem_euclid(period);
                            if p >= dim {
                                p = period - p;
                            }
                        }
                    }
                    2 => {
                        // edge: clamp to the nearest valid index
                        p = p.clamp(0, dim - 1);
                    }
                    m => panic!("Pad: unsupported mode {m}"),
                }
            }
            src += p as usize * xs[d];
        }
        if inside {
            out.dm()[oi * esz..(oi + 1) * esz].copy_from_slice(&x.d[src * esz..(src + 1) * esz]);
        }
    }
    out
}

/// ONNX Range: dtype follows the inputs (all three must share a dtype).
pub fn range(start: &T, limit: &T, delta: &T) -> T {
    let dt = start.dt;
    let (s, l, d) = (start.scalar_f(), limit.scalar_f(), delta.scalar_f());
    let n = (((l - s) / d).ceil()).max(0.0) as usize;
    match dt {
        Dt::I64 => {
            let (s, d) = (start.scalar_i(), delta.scalar_i());
            T::from_i64(vec![n], (0..n).map(|i| s + d * i as i64).collect())
        }
        Dt::I32 => {
            let (s, d) = (start.scalar_i(), delta.scalar_i());
            let v: Vec<i32> = (0..n).map(|i| (s + d * i as i64) as i32).collect();
            T::from_i32(vec![n], v)
        }
        Dt::F64 => {
            let (s, d) = (start.scalar_f64(), delta.scalar_f64());
            let mut out = T::new(Dt::F64, vec![n]);
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut f64, n) };
            for i in 0..n {
                o[i] = s + d * i as f64;
            }
            out
        }
        _ => {
            let v: Vec<f32> = (0..n).map(|i| s + d * i as f32).collect();
            T::from_f32(vec![n], v)
        }
    }
}

pub fn cumsum(x: &T, axis: i64) -> T {
    let ax = norm_axis(axis, x.dims.len());
    let inner: usize = x.dims[ax + 1..].iter().product();
    let dim = x.dims[ax];
    let outer = x.len() / (inner * dim);
    let xv = x.ints();
    if x.dt == Dt::I64 || x.dt == Dt::I32 {
        let dt = x.dt;
        let mut out = T::new(dt, x.dims.clone());
        let n = out.len();
        if dt == Dt::I64 {
            let ov =
                unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i64, n) };
            for o in 0..outer {
                for i in 0..inner {
                    let mut acc = 0i64;
                    for a in 0..dim {
                        let p = (o * dim + a) * inner + i;
                        acc += xv[p];
                        ov[p] = acc;
                    }
                }
            }
        } else {
            let ov =
                unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i32, n) };
            for o in 0..outer {
                for i in 0..inner {
                    let mut acc = 0i64;
                    for a in 0..dim {
                        let p = (o * dim + a) * inner + i;
                        acc += xv[p];
                        ov[p] = acc as i32;
                    }
                }
            }
        }
        return out;
    }
    let mut out = T::new(x.dt, x.dims.clone());
    let n = out.len();
    if x.dt == Dt::F64 {
        let xf: Vec<f64> =
            x.d.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
        let ov = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut f64, n) };
        for o in 0..outer {
            for i in 0..inner {
                let mut acc = 0f64;
                for a in 0..dim {
                    let p = (o * dim + a) * inner + i;
                    acc += xf[p];
                    ov[p] = acc;
                }
            }
        }
        return out;
    }
    let ov = out.fm();
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0f32;
            for a in 0..dim {
                let p = (o * dim + a) * inner + i;
                acc += xv[p] as f32;
                ov[p] = acc;
            }
        }
    }
    out
}

pub fn gather_elements(x: &T, idx: &T, axis: i64) -> T {
    let ax = norm_axis(axis, x.dims.len());
    let r = x.dims.len();
    let xs = strides(&x.dims);
    let is = strides(&idx.dims);
    let mut out = T::new(x.dt, idx.dims.clone());
    let iv: Vec<i64> = match idx.dt {
        Dt::I64 => idx.i64s().to_vec(),
        Dt::I32 => idx.i32s().iter().map(|&v| v as i64).collect(),
        _ => panic!("gather_elements idx"),
    };
    let mut c = vec![0usize; r];
    for oi in 0..out.len() {
        let mut rem = oi;
        for d in 0..r {
            c[d] = rem / is[d];
            rem %= is[d];
        }
        let mut xoff = 0usize;
        for d in 0..r {
            let p = if d == ax {
                let v = iv[oi];
                if v < 0 {
                    (v + x.dims[d] as i64) as usize
                } else {
                    v as usize
                }
            } else {
                c[d]
            };
            xoff += p * xs[d];
        }
        out.fm()[oi] = x.f()[xoff];
    }
    out
}

pub fn reduce_sum(x: &T, axes: &[i64], keep: bool) -> T {
    let r = x.dims.len();
    let ax: Vec<usize> = if axes.is_empty() {
        (0..r).collect()
    } else {
        axes.iter().map(|&a| norm_axis(a, r)).collect()
    };
    let mut od = Vec::new();
    for i in 0..r {
        if ax.contains(&i) {
            if keep {
                od.push(1);
            }
        } else {
            od.push(x.dims[i]);
        }
    }
    let mut out = T::new(Dt::F32, od.clone());
    let xs = strides(&x.dims);
    let os = strides(&od);
    let mut idx = vec![0usize; r];
    let ov = out.fm();
    let xv = x.f();
    let mut omap = vec![usize::MAX; r];
    {
        let mut oi = 0;
        for i in 0..r {
            if ax.contains(&i) {
                if keep {
                    omap[i] = oi;
                    oi += 1;
                }
            } else {
                omap[i] = oi;
                oi += 1;
            }
        }
    }
    for oi in 0..x.len() {
        let mut rem = oi;
        for d in 0..r {
            idx[d] = rem / xs[d];
            rem %= xs[d];
        }
        let mut doff = 0usize;
        for d in 0..r {
            if omap[d] != usize::MAX {
                doff += idx[d] * os[omap[d]];
            }
        }
        ov[doff] += xv[oi];
    }
    out
}
