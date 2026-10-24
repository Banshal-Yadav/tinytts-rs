#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let s = (h >> 15) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 0x1f {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    let b = f.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let ne = e - 127 + 15;
    if ne >= 0x1f {
        return s | 0x7c00;
    }
    if ne <= 0 {
        if ne < -10 {
            return s;
        }
        let mm = (m | 0x80_0000) >> (1 - ne);
        return s | ((mm >> 13) as u16);
    }
    s | ((ne as u16) << 10) | ((m >> 13) as u16)
}

pub fn dequant_i8(q: &[i8], scale: &[u16], zp: &[i8], out: &mut [f32], per_row: usize) {
    if scale.len() == 1 {
        let s = f16_to_f32(scale[0]);
        let z = zp[0] as f32;
        for (o, &v) in out.iter_mut().zip(q.iter()) {
            *o = (v as f32 - z) * s;
        }
    } else {
        for (r, chunk) in out.chunks_mut(per_row).enumerate() {
            let s = f16_to_f32(scale[r]);
            let z = zp[r] as f32;
            for (o, &v) in chunk.iter_mut().zip(q[r * per_row..].iter()) {
                *o = (v as f32 - z) * s;
            }
        }
    }
}

pub fn dequant_i8_f32(q: &[i8], scale: &[f32], zp: &[i8], out: &mut [f32], per_row: usize) {
    if scale.len() == 1 {
        let s = scale[0];
        let z = zp[0] as f32;
        for (o, &v) in out.iter_mut().zip(q.iter()) {
            *o = (v as f32 - z) * s;
        }
    } else {
        for (r, chunk) in out.chunks_mut(per_row).enumerate() {
            let s = scale[r];
            let z = zp[r] as f32;
            for (o, &v) in chunk.iter_mut().zip(q[r * per_row..].iter()) {
                *o = (v as f32 - z) * s;
            }
        }
    }
}

pub fn dequant_f16(q: &[u16], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(q.iter()) {
        *o = f16_to_f32(v);
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    #[inline]
    pub fn normal(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f32 + 0.5) / (1u64 << 53) as f32;
        let u2 = ((self.next_u64() >> 11) as f32 + 0.5) / (1u64 << 53) as f32;
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}
