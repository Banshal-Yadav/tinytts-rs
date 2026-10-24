//! Minimal `.npy` reader (v1.0/v2.0, C-order, little-endian).
//!
//! Only what the golden dumper produces: `<f4`, `<f2`, `<i8`, `<i4`, `<i2`,
//! `<u1`, `|b1`, `<f8`, `<i8` (int64). No dependencies.

// chunks_exact(N) is the idiomatic way to reinterpret a byte buffer as a
// fixed-width scalar; the suggested alternative is not clearer.
#![allow(clippy::chunks_exact_to_as_chunks)]
use std::fs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpDtype {
    F32,
    F16,
    F64,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl NpDtype {
    pub fn size(self) -> usize {
        match self {
            NpDtype::F32 | NpDtype::I32 => 4,
            NpDtype::F16 | NpDtype::I16 => 2,
            NpDtype::F64 | NpDtype::I64 => 8,
            NpDtype::I8 | NpDtype::U8 | NpDtype::Bool => 1,
        }
    }
}

pub struct Npy {
    pub dtype: NpDtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Npy {
    pub fn f32s(&self) -> Vec<f32> {
        match self.dtype {
            NpDtype::F32 => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            NpDtype::F16 => self
                .data
                .chunks_exact(2)
                .map(|c| tinytts::quant::f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect(),
            NpDtype::F64 => self
                .data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            NpDtype::I64 => self
                .data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            NpDtype::I32 => self
                .data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            NpDtype::I16 => self
                .data
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            NpDtype::I8 => self.data.iter().map(|&b| b as i8 as f32).collect(),
            NpDtype::U8 | NpDtype::Bool => self.data.iter().map(|&b| b as f32).collect(),
        }
    }
}

pub fn load(path: &str) -> Npy {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(bytes.len() > 10, "npy too short: {path}");
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "bad npy magic: {path}");

    let major = bytes[6];
    let (header_start, header_len) = if major == 1 {
        (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize)
    } else {
        (
            12usize,
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        )
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .expect("npy header utf8");

    let dtype = parse_dtype(header);
    let shape = parse_shape(header);
    let data = bytes[header_start + header_len..].to_vec();

    let expect: usize = shape.iter().product::<usize>() * dtype.size();
    assert_eq!(
        data.len(),
        expect,
        "npy {path}: data {} != expected {}",
        data.len(),
        expect
    );

    Npy { dtype, shape, data }
}

fn parse_dtype(header: &str) -> NpDtype {
    let i = header.find("'descr'").expect("descr");
    let rest = &header[i + "'descr'".len()..];
    // rest looks like:  : '<i8', 'fortran_order': ...
    let q1 = rest.find('\'').unwrap();
    let q2 = rest[q1 + 1..].find('\'').unwrap() + q1 + 1;
    let descr = &rest[q1 + 1..q2];
    match descr {
        "<f4" | "|f4" => NpDtype::F32,
        "<f2" | "|f2" => NpDtype::F16,
        "<f8" | "|f8" => NpDtype::F64,
        "<i8" | "|i8" => NpDtype::I64,
        "<i4" | "|i4" => NpDtype::I32,
        "<i2" | "|i2" => NpDtype::I16,
        "|i1" | "<i1" => NpDtype::I8,
        "|u1" | "<u1" => NpDtype::U8,
        "|b1" | "|?" => NpDtype::Bool,
        other => panic!("unsupported npy dtype {other}"),
    }
}

fn parse_shape(header: &str) -> Vec<usize> {
    let i = header.find("'shape'").expect("shape");
    let rest = &header[i..];
    let open = rest.find('(').unwrap();
    let close = rest.find(')').unwrap();
    let inner = &rest[open + 1..close];
    inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().expect("shape dim"))
        .collect()
}
