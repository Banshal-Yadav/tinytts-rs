use crate::onnx::Dt;
use std::sync::Arc;

/// Copy `src` into a fresh buffer whose allocation is 8-byte aligned.
///
/// The typed views below (`f`, `i64s`, `i32s`, `h`) reinterpret the byte buffer
/// as `f32`/`i64`/`i32`/`u16`. A `Vec<u8>` only guarantees alignment 1, so
/// building one with `to_vec()` and then reading it as `i64` is undefined
/// behaviour. Allocating through `Vec<u64>` gives alignment 8, which covers
/// every dtype the graph uses.
pub fn align_bytes(src: &[u8]) -> Vec<u8> {
    let n = src.len();
    let mut w: Vec<u64> = Vec::with_capacity(n.div_ceil(8).max(1));
    let mut i = 0;
    while i + 8 <= n {
        w.push(u64::from_le_bytes(src[i..i + 8].try_into().unwrap()));
        i += 8;
    }
    if i < n {
        let mut tail = [0u8; 8];
        tail[..n - i].copy_from_slice(&src[i..]);
        w.push(u64::from_le_bytes(tail));
    }
    let mut out = Vec::with_capacity((w.len() * 8).max(8));
    for x in w {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.truncate(n);
    out
}

/// A zeroed buffer with the same 8-byte alignment guarantee as `align_bytes`.
///
/// A zero-length `Vec<u8>` has a dangling pointer with alignment 1, which trips
/// the `from_raw_parts` precondition when the buffer is viewed as `i64`. Keep
/// the allocation at least one `u64` wide so the pointer stays 8-aligned.
fn zero_bytes(n: usize) -> Vec<u8> {
    let cap = n.div_ceil(8).max(1) * 8;
    let mut out = vec![0u8; cap];
    out.truncate(n);
    out
}

/// A tensor. `d` is refcounted so that views (`Reshape`, `Squeeze`, `Unsqueeze`,
/// `Identity`) and the initializer copies in `Exec::vals` are a pointer bump
/// instead of a full buffer copy. Mutation goes through `Arc::make_mut`, which
/// clones only when the buffer is shared.
#[derive(Clone, Debug)]
pub struct T {
    pub dt: Dt,
    pub dims: Vec<usize>,
    pub d: Arc<Vec<u8>>,
}

impl T {
    pub fn new(dt: Dt, dims: Vec<usize>) -> T {
        let n: usize = dims.iter().product();
        T {
            dt,
            dims,
            d: Arc::new(zero_bytes(n * dt.size())),
        }
    }
    pub fn from_f32(dims: Vec<usize>, v: Vec<f32>) -> T {
        T {
            dt: Dt::F32,
            dims,
            d: Arc::new(align_bytes(unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
            })),
        }
    }
    pub fn from_i64(dims: Vec<usize>, v: Vec<i64>) -> T {
        T {
            dt: Dt::I64,
            dims,
            d: Arc::new(align_bytes(unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
            })),
        }
    }
    pub fn from_i32(dims: Vec<usize>, v: Vec<i32>) -> T {
        T {
            dt: Dt::I32,
            dims,
            d: Arc::new(align_bytes(unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
            })),
        }
    }
    pub fn from_bool(dims: Vec<usize>, v: Vec<bool>) -> T {
        T {
            dt: Dt::Bool,
            dims,
            d: Arc::new(v.iter().map(|&b| b as u8).collect()),
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }
    #[inline]
    pub fn f(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.d.as_ptr() as *const f32, self.d.len() / 4) }
    }
    #[inline]
    pub fn fm(&mut self) -> &mut [f32] {
        let d = Arc::make_mut(&mut self.d);
        unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut f32, d.len() / 4) }
    }
    #[inline]
    pub fn h(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.d.as_ptr() as *const u16, self.d.len() / 2) }
    }
    #[inline]
    pub fn i8s(&self) -> &[i8] {
        unsafe { std::slice::from_raw_parts(self.d.as_ptr() as *const i8, self.d.len()) }
    }
    #[inline]
    pub fn i64s(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.d.as_ptr() as *const i64, self.d.len() / 8) }
    }
    #[inline]
    pub fn i32s(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.d.as_ptr() as *const i32, self.d.len() / 4) }
    }
    #[inline]
    pub fn bools(&self) -> &[u8] {
        &self.d
    }
    /// Mutable byte view. Clones the buffer only if it is shared.
    #[inline]
    pub fn dm(&mut self) -> &mut Vec<u8> {
        Arc::make_mut(&mut self.d)
    }
    pub fn scalar_f(&self) -> f32 {
        self.f()[0]
    }
    pub fn scalar_f64(&self) -> f64 {
        match self.dt {
            Dt::F64 => f64::from_le_bytes(self.d[..8].try_into().unwrap()),
            Dt::F32 => self.f()[0] as f64,
            Dt::I64 => self.i64s()[0] as f64,
            Dt::I32 => self.i32s()[0] as f64,
            _ => self.f()[0] as f64,
        }
    }
    pub fn ints(&self) -> Vec<i64> {
        match self.dt {
            Dt::I64 => self.i64s().to_vec(),
            Dt::I32 => self.i32s().iter().map(|&v| v as i64).collect(),
            Dt::F32 => self.f().iter().map(|&v| v as i64).collect(),
            Dt::F64 => self
                .d
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as i64)
                .collect(),
            Dt::F16 => self
                .h()
                .iter()
                .map(|&v| crate::quant::f16_to_f32(v) as i64)
                .collect(),
            Dt::I8 => self.i8s().iter().map(|&v| v as i64).collect(),
            Dt::U8 | Dt::Bool => self.d.iter().map(|&v| v as i64).collect(),
        }
    }
    pub fn scalar_i(&self) -> i64 {
        match self.dt {
            Dt::I64 => self.i64s()[0],
            Dt::I32 => self.i32s()[0] as i64,
            Dt::F32 => self.f()[0] as i64,
            _ => panic!("scalar_i on {:?}", self.dt),
        }
    }

    /// Dtype-aware f32 view, for numerical comparison in tests.
    pub fn f32s_for_compare(&self) -> Vec<f32> {
        match self.dt {
            Dt::F32 => self.f().to_vec(),
            Dt::F16 => self
                .h()
                .iter()
                .map(|&v| crate::quant::f16_to_f32(v))
                .collect(),
            Dt::F64 => self
                .d
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            Dt::I64 => self.i64s().iter().map(|&v| v as f32).collect(),
            Dt::I32 => self.i32s().iter().map(|&v| v as f32).collect(),
            Dt::I8 => self.i8s().iter().map(|&v| v as f32).collect(),
            Dt::U8 | Dt::Bool => self.d.iter().map(|&v| v as f32).collect(),
        }
    }
}

#[inline]
pub fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

#[inline]
pub fn norm_axis(a: i64, r: usize) -> usize {
    if a < 0 {
        (a + r as i64) as usize
    } else {
        a as usize
    }
}
