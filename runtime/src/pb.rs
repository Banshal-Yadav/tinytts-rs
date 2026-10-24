pub const VARINT: u8 = 0;
pub const I64: u8 = 1;
pub const LEN: u8 = 2;
pub const I32: u8 = 5;

#[inline]
pub fn varint(b: &[u8], p: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut s = 0;
    loop {
        if *p >= b.len() {
            panic!("varint overrun at {} of {}", p, b.len());
        }
        let c = b[*p];
        *p += 1;
        v |= ((c & 0x7f) as u64) << s;
        if c < 0x80 {
            return v;
        }
        s += 7;
    }
}

#[inline]
pub fn key(b: &[u8], p: &mut usize) -> (u32, u8) {
    let k = varint(b, p);
    ((k >> 3) as u32, (k & 7) as u8)
}

#[inline]
pub fn skip(b: &[u8], p: &mut usize, wt: u8) {
    match wt {
        VARINT => {
            varint(b, p);
        }
        I64 => *p += 8,
        LEN => {
            let n = varint(b, p) as usize;
            *p += n;
        }
        I32 => *p += 4,
        3 => loop {
            if *p >= b.len() {
                panic!("sgroup overrun");
            }
            let (_, w2) = key(b, p);
            if w2 == 4 {
                break;
            }
        },
        4 => {}
        _ => panic!("bad wire type {wt}"),
    }
}

#[inline]
pub fn bytes<'a>(b: &'a [u8], p: &mut usize) -> &'a [u8] {
    let n = varint(b, p) as usize;
    let s = &b[*p..*p + n];
    *p += n;
    s
}

#[inline]
pub fn f32le(b: &[u8], p: &mut usize) -> f32 {
    let v = f32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}

#[inline]
pub fn i32le(b: &[u8], p: &mut usize) -> i32 {
    let v = i32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}

#[inline]
pub fn i64le(b: &[u8], p: &mut usize) -> i64 {
    let v = i64::from_le_bytes(b[*p..*p + 8].try_into().unwrap());
    *p += 8;
    v
}

pub fn f32s(b: &[u8], p: &mut usize) -> Vec<f32> {
    let n = varint(b, p) as usize;
    let s = &b[*p..*p + n];
    *p += n;
    s.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn i64s(b: &[u8], p: &mut usize) -> Vec<i64> {
    let n = varint(b, p) as usize;
    let s = &b[*p..*p + n];
    *p += n;
    let mut q = 0;
    let mut v = Vec::new();
    while q < s.len() {
        v.push(varint(s, &mut q) as i64);
    }
    v
}

pub fn i32s(b: &[u8], p: &mut usize) -> Vec<i32> {
    let n = varint(b, p) as usize;
    let s = &b[*p..*p + n];
    *p += n;
    s.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
