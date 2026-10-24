//! Tiny JSON reader for the golden manifest. No dependencies.
//!
//! Only handles the subset `dump_golden.py` emits: objects, arrays, strings,
//! numbers, bools, null. Enough to pull out node order and tensor metadata.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum J {
    Null,
    Bool,
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn get(&self, key: &str) -> Option<&J> {
        match self {
            J::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &str {
        match self {
            J::Str(s) => s,
            _ => panic!("not a string: {self:?}"),
        }
    }
    pub fn as_arr(&self) -> &[J] {
        match self {
            J::Arr(a) => a,
            _ => panic!("not an array: {self:?}"),
        }
    }
    pub fn as_usize(&self) -> usize {
        match self {
            J::Num(n) => *n as usize,
            _ => panic!("not a number: {self:?}"),
        }
    }
    pub fn as_obj(&self) -> &[(String, J)] {
        match self {
            J::Obj(kv) => kv,
            _ => panic!("not an object: {self:?}"),
        }
    }
}

pub fn parse(s: &str) -> J {
    let b = s.as_bytes();
    let mut p = 0;
    value(b, &mut p)
}

fn ws(b: &[u8], p: &mut usize) {
    while *p < b.len() && matches!(b[*p], b' ' | b'\t' | b'\n' | b'\r') {
        *p += 1;
    }
}

fn value(b: &[u8], p: &mut usize) -> J {
    ws(b, p);
    match b[*p] {
        b'{' => obj(b, p),
        b'[' => arr(b, p),
        b'"' => J::Str(string(b, p)),
        b't' => {
            *p += 4;
            J::Bool
        }
        b'f' => {
            *p += 5;
            J::Bool
        }
        b'n' => {
            *p += 4;
            J::Null
        }
        _ => num(b, p),
    }
}

fn obj(b: &[u8], p: &mut usize) -> J {
    *p += 1; // {
    let mut kv = Vec::new();
    loop {
        ws(b, p);
        if b[*p] == b'}' {
            *p += 1;
            break;
        }
        let k = string(b, p);
        ws(b, p);
        assert_eq!(b[*p], b':');
        *p += 1;
        let v = value(b, p);
        kv.push((k, v));
        ws(b, p);
        if b[*p] == b',' {
            *p += 1;
        }
    }
    J::Obj(kv)
}

fn arr(b: &[u8], p: &mut usize) -> J {
    *p += 1; // [
    let mut a = Vec::new();
    loop {
        ws(b, p);
        if b[*p] == b']' {
            *p += 1;
            break;
        }
        a.push(value(b, p));
        ws(b, p);
        if b[*p] == b',' {
            *p += 1;
        }
    }
    J::Arr(a)
}

fn string(b: &[u8], p: &mut usize) -> String {
    ws(b, p);
    assert_eq!(b[*p], b'"');
    *p += 1;
    let mut s = String::new();
    while b[*p] != b'"' {
        if b[*p] == b'\\' {
            *p += 1;
            match b[*p] {
                b'n' => s.push('\n'),
                b't' => s.push('\t'),
                b'r' => s.push('\r'),
                b'"' => s.push('"'),
                b'\\' => s.push('\\'),
                b'/' => s.push('/'),
                b'u' => {
                    let hex = std::str::from_utf8(&b[*p + 1..*p + 5]).unwrap();
                    let cp = u32::from_str_radix(hex, 16).unwrap();
                    s.push(char::from_u32(cp).unwrap_or('?'));
                    *p += 4;
                }
                other => s.push(other as char),
            }
            *p += 1;
        } else {
            s.push(b[*p] as char);
            *p += 1;
        }
    }
    *p += 1;
    s
}

fn num(b: &[u8], p: &mut usize) -> J {
    let start = *p;
    while *p < b.len() && matches!(b[*p], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
        *p += 1;
    }
    J::Num(std::str::from_utf8(&b[start..*p]).unwrap().parse().unwrap())
}

/// Convenience: tensor name -> file name.
pub struct TensorMeta {
    pub file: String,
}

pub fn tensors(manifest: &J) -> HashMap<String, TensorMeta> {
    let mut out = HashMap::new();
    for (name, v) in manifest.get("tensors").unwrap().as_obj() {
        out.insert(
            name.clone(),
            TensorMeta {
                file: v.get("file").unwrap().as_str().to_string(),
            },
        );
    }
    out
}
