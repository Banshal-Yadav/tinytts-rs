use crate::onnx::Dt;
use crate::tensor::T;

/// Result dtype for a binary op, following ONNX type rules.
///
/// Comparisons (Equal/Less/Greater/GreaterOrEqual) always yield bool.
/// Arithmetic yields the "widest" of the two input dtypes, with f64 > f32 >
/// i64 > i32 > i8/u8. Mixed int/float promotes to float.
fn binop_out_dtype(a: Dt, b: Dt, op: u8) -> Dt {
    // 7..=11 are Equal, Greater, Less, GreaterOrEqual, LessOrEqual.
    if (7..=11).contains(&op) {
        return Dt::Bool;
    }
    let rank = |d: Dt| -> u8 {
        match d {
            Dt::F64 => 6,
            Dt::F32 => 5,
            Dt::F16 => 4,
            Dt::I64 => 3,
            Dt::I32 => 2,
            Dt::I8 | Dt::U8 | Dt::Bool => 1,
        }
    };
    if rank(a) >= rank(b) {
        a
    } else {
        b
    }
}

pub fn binary(a: &T, b: &T, op: u8) -> T {
    let dt = binop_out_dtype(a.dt, b.dt, op);
    // Fast paths for the shapes that dominate the profile. All three require
    // f32 inputs and an f32 result, which is what the hot nodes are.
    if dt == Dt::F32 && a.dt == Dt::F32 && b.dt == Dt::F32 {
        if let Some(r) = binary_f32_fast(a, b, op) {
            return r;
        }
    }
    binary_general(a, b, op, dt)
}

/// Same-shape, scalar, and last-dim-broadcast f32 paths.
///
/// Returns `None` when the shapes need the general strided path.
fn binary_f32_fast(a: &T, b: &T, op: u8) -> Option<T> {
    let av = a.f();
    let bv = b.f();
    // same shape
    if a.dims == b.dims {
        let mut out = T::new(Dt::F32, a.dims.clone());
        let ov = out.fm();
        apply_f32(ov, av, bv, op);
        return Some(out);
    }
    // scalar (either side)
    if b.len() == 1 {
        let mut out = T::new(Dt::F32, a.dims.clone());
        let ov = out.fm();
        let s = bv[0];
        for (o, &x) in ov.iter_mut().zip(av.iter()) {
            *o = scalar_op(x, s, op);
        }
        return Some(out);
    }
    if a.len() == 1 {
        let mut out = T::new(Dt::F32, b.dims.clone());
        let ov = out.fm();
        let s = av[0];
        for (o, &y) in ov.iter_mut().zip(bv.iter()) {
            *o = scalar_op(s, y, op);
        }
        return Some(out);
    }
    // last-dim broadcast: b is [.., 1] or [1, .., 1] against a's trailing dim
    if a.dims.len() == b.dims.len() && a.dims[..a.dims.len() - 1] == b.dims[..b.dims.len() - 1] {
        let inner = *a.dims.last().unwrap();
        if b.dims[b.dims.len() - 1] == 1 && inner > 1 {
            let mut out = T::new(Dt::F32, a.dims.clone());
            let ov = out.fm();
            for (ci, (oc, ob)) in ov.chunks_exact_mut(inner).zip(bv.iter()).enumerate() {
                let s = *ob;
                let src = &av[ci * inner..ci * inner + inner];
                for (o, &x) in oc.iter_mut().zip(src.iter()) {
                    *o = scalar_op(x, s, op);
                }
            }
            return Some(out);
        }
    }
    // per-channel: b is [1, C, 1] against a [1, C, L]
    if a.dims.len() == 3
        && b.dims.len() == 3
        && b.dims[0] == 1
        && b.dims[2] == 1
        && a.dims[1] == b.dims[1]
    {
        let (c, l) = (a.dims[1], a.dims[2]);
        let mut out = T::new(Dt::F32, a.dims.clone());
        let ov = out.fm();
        for ci in 0..c {
            let s = bv[ci];
            let src = &av[ci * l..ci * l + l];
            let dst = &mut ov[ci * l..ci * l + l];
            for (o, &x) in dst.iter_mut().zip(src.iter()) {
                *o = scalar_op(x, s, op);
            }
        }
        return Some(out);
    }
    // outer product: a is [1, 1, L], b is [1, C, 1] -> [1, C, L]
    if a.dims.len() == 3
        && b.dims.len() == 3
        && a.dims[0] == 1
        && a.dims[1] == 1
        && b.dims[0] == 1
        && b.dims[2] == 1
    {
        let (c, l) = (b.dims[1], a.dims[2]);
        let mut out = T::new(Dt::F32, vec![1, c, l]);
        let ov = out.fm();
        for ci in 0..c {
            let s = bv[ci];
            let dst = &mut ov[ci * l..ci * l + l];
            for (o, &x) in dst.iter_mut().zip(av.iter()) {
                *o = scalar_op(x, s, op);
            }
        }
        return Some(out);
    }
    // outer product, mirrored: a is [1, C, 1], b is [1, 1, L] -> [1, C, L]
    if a.dims.len() == 3
        && b.dims.len() == 3
        && a.dims[0] == 1
        && a.dims[2] == 1
        && b.dims[0] == 1
        && b.dims[1] == 1
    {
        let (c, l) = (a.dims[1], b.dims[2]);
        let mut out = T::new(Dt::F32, vec![1, c, l]);
        let ov = out.fm();
        for ci in 0..c {
            let s = av[ci];
            let dst = &mut ov[ci * l..ci * l + l];
            for (o, &y) in dst.iter_mut().zip(bv.iter()) {
                *o = scalar_op(s, y, op);
            }
        }
        return Some(out);
    }
    // per-channel: b is [1, C, 1] against a [1, C, L] where a is 3-D but b 3-D
    if a.dims.len() == 3
        && b.dims.len() == 3
        && b.dims[0] == 1
        && b.dims[1] == 1
        && a.dims[2] == b.dims[2]
    {
        let (c, l) = (a.dims[1], a.dims[2]);
        let mut out = T::new(Dt::F32, a.dims.clone());
        let ov = out.fm();
        for ci in 0..c {
            let src = &av[ci * l..ci * l + l];
            let dst = &mut ov[ci * l..ci * l + l];
            apply_f32(dst, src, bv, op);
        }
        return Some(out);
    }
    None
}

#[inline]
fn scalar_op(x: f32, y: f32, op: u8) -> f32 {
    match op {
        0 => x + y,
        1 => x - y,
        2 => x * y,
        3 => x / y,
        4 => x.max(y),
        5 => x.min(y),
        6 => x.powf(y),
        12 => x.atan2(y),
        _ => panic!("binop {op}"),
    }
}

#[inline]
fn apply_f32(o: &mut [f32], a: &[f32], b: &[f32], op: u8) {
    match op {
        0 => {
            for i in 0..o.len() {
                o[i] = a[i] + b[i];
            }
        }
        1 => {
            for i in 0..o.len() {
                o[i] = a[i] - b[i];
            }
        }
        2 => {
            for i in 0..o.len() {
                o[i] = a[i] * b[i];
            }
        }
        3 => {
            for i in 0..o.len() {
                o[i] = a[i] / b[i];
            }
        }
        4 => {
            for i in 0..o.len() {
                o[i] = a[i].max(b[i]);
            }
        }
        5 => {
            for i in 0..o.len() {
                o[i] = a[i].min(b[i]);
            }
        }
        6 => {
            for i in 0..o.len() {
                o[i] = a[i].powf(b[i]);
            }
        }
        12 => {
            for i in 0..o.len() {
                o[i] = a[i].atan2(b[i]);
            }
        }
        _ => panic!("binop {op}"),
    }
}

fn binary_general(a: &T, b: &T, op: u8, dt: Dt) -> T {
    let (od, av, bv) = broadcast(a, b);
    let mut out = T::new(dt, od);
    let n = out.len();
    match dt {
        Dt::Bool => {
            for i in 0..n {
                let x = av(i);
                let y = bv(i);
                out.dm()[i] = match op {
                    7 => (x == y) as u8,
                    8 => (x > y) as u8,
                    9 => (x < y) as u8,
                    10 => (x >= y) as u8,
                    11 => (x <= y) as u8,
                    _ => unreachable!(),
                };
            }
        }
        Dt::I64 => {
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i64, n) };
            for i in 0..n {
                let x = av(i) as i64;
                let y = bv(i) as i64;
                o[i] = match op {
                    0 => x + y,
                    1 => x - y,
                    2 => x * y,
                    3 => {
                        if y != 0 {
                            x / y
                        } else {
                            0
                        }
                    }
                    4 => x.max(y),
                    5 => x.min(y),
                    6 => (x as f64).powf(y as f64) as i64,
                    12 => (x as f64).atan2(y as f64) as i64,
                    _ => panic!("binop {op} on i64"),
                };
            }
        }
        Dt::I32 => {
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i32, n) };
            for i in 0..n {
                let x = av(i) as i32;
                let y = bv(i) as i32;
                o[i] = match op {
                    0 => x + y,
                    1 => x - y,
                    2 => x * y,
                    3 => {
                        if y != 0 {
                            x / y
                        } else {
                            0
                        }
                    }
                    4 => x.max(y),
                    5 => x.min(y),
                    6 => (x as f64).powf(y as f64) as i32,
                    12 => (x as f64).atan2(y as f64) as i32,
                    _ => panic!("binop {op} on i32"),
                };
            }
        }
        Dt::F64 => {
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut f64, n) };
            for i in 0..n {
                let x = av(i) as f64;
                let y = bv(i) as f64;
                o[i] = match op {
                    0 => x + y,
                    1 => x - y,
                    2 => x * y,
                    3 => x / y,
                    4 => x.max(y),
                    5 => x.min(y),
                    6 => x.powf(y),
                    12 => x.atan2(y),
                    _ => panic!("binop {op} on f64"),
                };
            }
        }
        Dt::F16 => {
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut u16, n) };
            for i in 0..n {
                let x = av(i);
                let y = bv(i);
                let v = match op {
                    0 => x + y,
                    1 => x - y,
                    2 => x * y,
                    3 => x / y,
                    4 => x.max(y),
                    5 => x.min(y),
                    6 => x.powf(y),
                    12 => x.atan2(y),
                    _ => panic!("binop {op} on f16"),
                };
                o[i] = crate::quant::f32_to_f16(v);
            }
        }
        _ => {
            let ov = out.fm();
            for i in 0..n {
                let x = av(i);
                let y = bv(i);
                ov[i] = match op {
                    0 => x + y,
                    1 => x - y,
                    2 => x * y,
                    3 => x / y,
                    4 => x.max(y),
                    5 => x.min(y),
                    6 => x.powf(y),
                    12 => x.atan2(y),
                    _ => panic!("binop {op}"),
                };
            }
        }
    }
    out
}

pub fn broadcast<'a>(
    a: &'a T,
    b: &'a T,
) -> (
    Vec<usize>,
    impl Fn(usize) -> f32 + 'a,
    impl Fn(usize) -> f32 + 'a,
) {
    let r = a.dims.len().max(b.dims.len());
    let mut od = vec![0usize; r];
    let mut ad = vec![1usize; r];
    let mut bd = vec![1usize; r];
    for i in 0..a.dims.len() {
        ad[r - a.dims.len() + i] = a.dims[i];
    }
    for i in 0..b.dims.len() {
        bd[r - b.dims.len() + i] = b.dims[i];
    }
    for i in 0..r {
        od[i] = ad[i].max(bd[i]);
    }
    let os = crate::tensor::strides(&od);
    let os2 = os.clone();
    let as_ = crate::tensor::strides(&ad);
    let bs = crate::tensor::strides(&bd);
    let av = a.f();
    let bv = b.f();
    let adt = a.dt;
    let bdt = b.dt;
    let araw = a.d.as_slice();
    let braw = b.d.as_slice();
    let aoff = move |i: usize| -> usize {
        let os = &os2;
        let mut rem = i;
        let mut off = 0usize;
        for d in 0..r {
            let p = rem / os[d];
            rem %= os[d];
            if ad[d] != 1 {
                off += p * as_[d];
            }
        }
        off
    };
    let boff = move |i: usize| -> usize {
        let mut rem = i;
        let mut off = 0usize;
        for d in 0..r {
            let p = rem / os[d];
            rem %= os[d];
            if bd[d] != 1 {
                off += p * bs[d];
            }
        }
        off
    };
    (
        od,
        move |i| val(adt, araw, av, aoff(i)),
        move |i| val(bdt, braw, bv, boff(i)),
    )
}

#[inline]
fn val(dt: Dt, raw: &[u8], fv: &[f32], i: usize) -> f32 {
    match dt {
        Dt::F32 => fv[i],
        Dt::I64 => i64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap()) as f32,
        Dt::I32 => i32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()) as f32,
        Dt::F16 => crate::quant::f16_to_f32(u16::from_le_bytes(
            raw[i * 2..i * 2 + 2].try_into().unwrap(),
        )),
        Dt::I8 => raw[i] as i8 as f32,
        Dt::U8 | Dt::Bool => raw[i] as f32,
        Dt::F64 => f64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap()) as f32,
    }
}

pub fn unary(x: &T, op: u8) -> T {
    let n = x.len();
    // Floor/Ceil/Round/Neg/Abs preserve integer dtypes; everything else is float.
    let int_preserving = matches!(op, 0 | 8 | 9 | 10 | 11);
    if int_preserving && matches!(x.dt, Dt::I64 | Dt::I32) {
        let mut out = T::new(x.dt, x.dims.clone());
        let xv = x.ints();
        if x.dt == Dt::I64 {
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i64, n) };
            for i in 0..n {
                o[i] = match op {
                    0 => -xv[i],
                    8 => xv[i].abs(),
                    _ => xv[i],
                };
            }
        } else {
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i32, n) };
            for i in 0..n {
                o[i] = match op {
                    0 => -(xv[i] as i32),
                    8 => (xv[i] as i32).abs(),
                    _ => xv[i] as i32,
                };
            }
        }
        return out;
    }
    if x.dt == Dt::F64 {
        let mut out = T::new(Dt::F64, x.dims.clone());
        let xv: Vec<f64> =
            x.d.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
        let ov = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut f64, n) };
        for i in 0..n {
            let v = xv[i];
            ov[i] = match op {
                0 => -v,
                1 => v.exp(),
                2 => v.ln(),
                3 => v.sqrt(),
                4 => v.tanh(),
                5 => 1.0 / (1.0 + (-v).exp()),
                6 => v.sin(),
                7 => v.cos(),
                8 => v.abs(),
                9 => v.floor(),
                10 => v.ceil(),
                11 => v.round(),
                12 => crate::quant::erf(v as f32) as f64,
                13 => v.recip(),
                _ => panic!("unop {op}"),
            };
        }
        return out;
    }
    let mut out = T::new(Dt::F32, x.dims.clone());
    let xv = x.f();
    let ov = out.fm();
    for i in 0..ov.len() {
        let v = xv[i];
        ov[i] = match op {
            0 => -v,
            1 => v.exp(),
            2 => v.ln(),
            3 => v.sqrt(),
            4 => v.tanh(),
            5 => crate::quant::sigmoid(v),
            6 => v.sin(),
            7 => v.cos(),
            8 => v.abs(),
            9 => v.floor(),
            10 => v.ceil(),
            11 => v.round(),
            12 => crate::quant::erf(v),
            13 => v.recip(),
            _ => panic!("unop {op}"),
        };
    }
    out
}

pub fn cast(x: &T, to: Dt) -> T {
    let n = x.len();
    let mut out = T::new(to, x.dims.clone());
    match (x.dt, to) {
        (Dt::F32, Dt::F32) => {
            return T {
                dt: to,
                dims: x.dims.clone(),
                d: x.d.clone(),
            }
        }
        (Dt::F16, Dt::F32) => {
            let s = x.h();
            let o = out.fm();
            for i in 0..n {
                o[i] = crate::quant::f16_to_f32(s[i]);
            }
        }
        (Dt::F32, Dt::F16) => {
            let s = x.f();
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut u16, n) };
            for i in 0..n {
                o[i] = crate::quant::f32_to_f16(s[i]);
            }
        }
        (Dt::I64, Dt::F32) => {
            let s = x.i64s();
            let o = out.fm();
            for i in 0..n {
                o[i] = s[i] as f32;
            }
        }
        (Dt::I32, Dt::F32) => {
            let s = x.i32s();
            let o = out.fm();
            for i in 0..n {
                o[i] = s[i] as f32;
            }
        }
        (Dt::F32, Dt::I64) => {
            let s = x.f();
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i64, n) };
            for i in 0..n {
                o[i] = s[i] as i64;
            }
        }
        (Dt::F32, Dt::I32) => {
            let s = x.f();
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i32, n) };
            for i in 0..n {
                o[i] = s[i] as i32;
            }
        }
        (Dt::I64, Dt::I32) => {
            let s = x.i64s();
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i32, n) };
            for i in 0..n {
                o[i] = s[i] as i32;
            }
        }
        (Dt::I32, Dt::I64) => {
            let s = x.i32s();
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i64, n) };
            for i in 0..n {
                o[i] = s[i] as i64;
            }
        }
        (Dt::Bool, Dt::F32) => {
            let s = x.bools();
            let o = out.fm();
            for i in 0..n {
                o[i] = s[i] as f32;
            }
        }
        (Dt::F32, Dt::Bool) => {
            let s = x.f();
            for i in 0..n {
                out.dm()[i] = (s[i] != 0.0) as u8;
            }
        }
        (Dt::I64, Dt::Bool) => {
            let s = x.i64s();
            for i in 0..n {
                out.dm()[i] = (s[i] != 0) as u8;
            }
        }
        (Dt::I32, Dt::Bool) => {
            let s = x.i32s();
            for i in 0..n {
                out.dm()[i] = (s[i] != 0) as u8;
            }
        }
        (Dt::Bool, Dt::I64) => {
            let s = x.bools();
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i64, n) };
            for i in 0..n {
                o[i] = s[i] as i64;
            }
        }
        (Dt::Bool, Dt::I32) => {
            let s = x.bools();
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i32, n) };
            for i in 0..n {
                o[i] = s[i] as i32;
            }
        }
        (Dt::I8, Dt::F32) => {
            let s = x.i8s();
            let o = out.fm();
            for i in 0..n {
                o[i] = s[i] as f32;
            }
        }
        (Dt::F32, Dt::F64) => {
            let s = x.f();
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut f64, n) };
            for i in 0..n {
                o[i] = s[i] as f64;
            }
        }
        (Dt::F64, Dt::F32) => {
            let s: Vec<f64> =
                x.d.chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
            let o = out.fm();
            for i in 0..n {
                o[i] = s[i] as f32;
            }
        }
        (Dt::I64, Dt::F64) => {
            let s = x.i64s();
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut f64, n) };
            for i in 0..n {
                o[i] = s[i] as f64;
            }
        }
        (Dt::F64, Dt::I64) => {
            let s: Vec<f64> =
                x.d.chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i64, n) };
            for i in 0..n {
                o[i] = s[i] as i64;
            }
        }
        (a, b) if a == b => return x.clone(),
        (a, b) => panic!("cast {a:?}->{b:?}"),
    }
    out
}

pub fn where_(c: &T, a: &T, b: &T) -> T {
    let (od, cv, _) = broadcast(c, c);
    let (od2, av, bv) = broadcast(a, b);
    let od = if od2.len() > od.len() { od2 } else { od };
    let dt = if a.dt == b.dt { a.dt } else { Dt::F32 };
    let mut out = T::new(dt, od);
    let n = out.len();
    let cvals = c.ints();
    let _ = cv;
    match dt {
        Dt::I64 => {
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i64, n) };
            for i in 0..n {
                o[i] = if cvals[i] != 0 {
                    av(i) as i64
                } else {
                    bv(i) as i64
                };
            }
        }
        Dt::I32 => {
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i32, n) };
            for i in 0..n {
                o[i] = if cvals[i] != 0 {
                    av(i) as i32
                } else {
                    bv(i) as i32
                };
            }
        }
        _ => {
            let o = out.fm();
            for i in 0..n {
                o[i] = if cvals[i] != 0 { av(i) } else { bv(i) };
            }
        }
    }
    out
}

pub fn clip(x: &T, lo: f32, hi: f32) -> T {
    let mut out = T::new(Dt::F32, x.dims.clone());
    let xv = x.f();
    let ov = out.fm();
    for i in 0..ov.len() {
        ov[i] = xv[i].clamp(lo, hi);
    }
    out
}

pub fn leaky(x: &T, alpha: f32) -> T {
    let mut out = T::new(Dt::F32, x.dims.clone());
    let xv = x.f();
    let ov = out.fm();
    for i in 0..ov.len() {
        let v = xv[i];
        ov[i] = if v >= 0.0 { v } else { alpha * v };
    }
    out
}

pub fn isnan(x: &T) -> T {
    let mut out = T::new(Dt::Bool, x.dims.clone());
    let xv = x.f();
    for i in 0..xv.len() {
        out.dm()[i] = xv[i].is_nan() as u8;
    }
    out
}

pub fn logical(a: &T, b: &T, op: u8) -> T {
    let (od, av, bv) = broadcast(a, b);
    let mut out = T::new(Dt::Bool, od);
    for i in 0..out.d.len() {
        let x = av(i) != 0.0;
        let y = bv(i) != 0.0;
        out.dm()[i] = match op {
            0 => (x && y) as u8,
            1 => (x || y) as u8,
            _ => panic!(),
        };
    }
    out
}

pub fn mod_(a: &T, b: &T, fmod: bool) -> T {
    let (od, av, bv) = broadcast(a, b);
    let dt = if a.dt == b.dt { a.dt } else { Dt::F32 };
    let mut out = T::new(dt, od);
    let n = out.len();
    match dt {
        Dt::I64 => {
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i64, n) };
            for i in 0..n {
                let x = av(i) as i64;
                let y = bv(i) as i64;
                o[i] = if y != 0 { x % y } else { 0 };
            }
        }
        Dt::I32 => {
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i32, n) };
            for i in 0..n {
                let x = av(i) as i32;
                let y = bv(i) as i32;
                o[i] = if y != 0 { x % y } else { 0 };
            }
        }
        Dt::F64 => {
            let o = unsafe { std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut f64, n) };
            for i in 0..n {
                let x = av(i) as f64;
                let y = bv(i) as f64;
                o[i] = if fmod { x % y } else { x - y * (x / y).floor() };
            }
        }
        _ => {
            let ov = out.fm();
            for i in 0..n {
                let x = av(i);
                let y = bv(i);
                ov[i] = if fmod { x % y } else { x - y * (x / y).floor() };
            }
        }
    }
    out
}
