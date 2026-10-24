use crate::pb::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dt {
    F32,
    F16,
    I8,
    U8,
    I32,
    I64,
    F64,
    Bool,
}

impl Dt {
    pub fn from_onnx(v: i32) -> Dt {
        match v {
            1 => Dt::F32,
            10 => Dt::F16,
            3 => Dt::I8,
            2 => Dt::U8,
            6 => Dt::I32,
            7 => Dt::I64,
            11 => Dt::F64,
            9 => Dt::Bool,
            _ => panic!("dtype {v}"),
        }
    }
    #[inline]
    pub fn size(self) -> usize {
        match self {
            Dt::F32 | Dt::I32 => 4,
            Dt::F16 => 2,
            Dt::I8 | Dt::U8 | Dt::Bool => 1,
            Dt::I64 | Dt::F64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub dt: Dt,
    pub dims: Vec<usize>,
    pub data: std::sync::Arc<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub enum Attr {
    F(f32),
    I(i64),
    S(Vec<u8>),
    T(Vec<u8>),
    Fs(Vec<f32>),
    Is(Vec<i64>),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: String,
    pub name: String,
    pub ins: Vec<String>,
    pub outs: Vec<String>,
    pub attrs: Vec<(String, Attr)>,
}

impl Node {
    pub fn attr_i(&self, k: &str) -> Option<i64> {
        self.attrs
            .iter()
            .find(|(n, _)| n == k)
            .and_then(|(_, a)| match a {
                Attr::I(v) => Some(*v),
                _ => None,
            })
    }
    pub fn attr_f(&self, k: &str) -> Option<f32> {
        self.attrs
            .iter()
            .find(|(n, _)| n == k)
            .and_then(|(_, a)| match a {
                Attr::F(v) => Some(*v),
                _ => None,
            })
    }
    pub fn attr_is(&self, k: &str) -> Option<&[i64]> {
        self.attrs
            .iter()
            .find(|(n, _)| n == k)
            .and_then(|(_, a)| match a {
                Attr::Is(v) => Some(v.as_slice()),
                _ => None,
            })
    }
    pub fn attr_s(&self, k: &str) -> Option<&[u8]> {
        self.attrs
            .iter()
            .find(|(n, _)| n == k)
            .and_then(|(_, a)| match a {
                Attr::S(v) => Some(v.as_slice()),
                _ => None,
            })
    }
}

pub struct Graph {
    pub nodes: Vec<Node>,
    pub inits: Vec<(String, Tensor)>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

pub fn parse_tensor(b: &[u8]) -> Tensor {
    let mut p = 0;
    let mut dt = Dt::F32;
    let mut dims = Vec::new();
    let mut raw: &[u8] = &[];
    let mut f32v: Vec<f32> = Vec::new();
    let mut i64v: Vec<i64> = Vec::new();
    let mut i32v: Vec<i32> = Vec::new();
    while p < b.len() {
        let (f, wt) = key(b, &mut p);
        match (f, wt) {
            (1, LEN) => dims = i64s(b, &mut p).into_iter().map(|v| v as usize).collect(),
            (1, VARINT) => dims.push(varint(b, &mut p) as usize),
            (2, VARINT) => dt = Dt::from_onnx(varint(b, &mut p) as i32),
            (4, LEN) => f32v = f32s(b, &mut p),
            (4, I32) => f32v.push(f32le(b, &mut p)),
            (5, LEN) => i32v = i32s(b, &mut p),
            (5, I32) => i32v.push(i32le(b, &mut p)),
            (7, LEN) => i64v = i64s(b, &mut p),
            (7, VARINT) => i64v.push(varint(b, &mut p) as i64),
            (8, LEN) => {
                let mut q = p;
                let n = varint(b, &mut q) as usize;
                p = q + n;
            }
            (9, LEN) => raw = bytes(b, &mut p),
            (10, LEN) | (11, LEN) | (12, LEN) => {
                let mut q = p;
                let n = varint(b, &mut q) as usize;
                p = q + n;
            }
            _ => skip(b, &mut p, wt),
        }
    }
    let data = if !raw.is_empty() {
        raw.to_vec()
    } else {
        match dt {
            Dt::F32 => f32v.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Dt::I64 => i64v.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Dt::I32 => i32v.iter().flat_map(|v| v.to_le_bytes()).collect(),
            _ => panic!("tensor {dt:?} without raw_data"),
        }
    };
    // `Tensor.data` is viewed as `f32`/`i64`/`f64` by `tensor.rs`, so it must be
    // 8-byte aligned. `raw_data` is a byte string with no alignment guarantee,
    // and an empty tensor's `Vec<u8>` is dangling with alignment 1.
    Tensor {
        dt,
        dims,
        data: std::sync::Arc::new(crate::tensor::align_bytes(&data)),
    }
}

fn parse_attr(b: &[u8]) -> (String, Attr) {
    let mut p = 0;
    let mut name = String::new();
    let mut val = Attr::I(0);
    while p < b.len() {
        let (f, wt) = key(b, &mut p);
        match (f, wt) {
            (1, LEN) => name = String::from_utf8_lossy(bytes(b, &mut p)).into_owned(),
            (2, I32) => val = Attr::F(f32le(b, &mut p)),
            (3, VARINT) => val = Attr::I(varint(b, &mut p) as i64),
            (4, LEN) => val = Attr::S(bytes(b, &mut p).to_vec()),
            (5, LEN) => val = Attr::T(bytes(b, &mut p).to_vec()),
            (7, LEN) => val = Attr::Fs(f32s(b, &mut p)),
            (7, I32) => match &mut val {
                Attr::Fs(v) => v.push(f32le(b, &mut p)),
                _ => val = Attr::Fs(vec![f32le(b, &mut p)]),
            },
            (8, LEN) => val = Attr::Is(i64s(b, &mut p)),
            (8, VARINT) => match &mut val {
                Attr::Is(v) => v.push(varint(b, &mut p) as i64),
                _ => val = Attr::Is(vec![varint(b, &mut p) as i64]),
            },
            _ => skip(b, &mut p, wt),
        }
    }
    (name, val)
}

fn parse_node(b: &[u8]) -> Node {
    let mut p = 0;
    let mut n = Node {
        op: String::new(),
        name: String::new(),
        ins: Vec::new(),
        outs: Vec::new(),
        attrs: Vec::new(),
    };
    while p < b.len() {
        let (f, wt) = key(b, &mut p);
        match f {
            1 => n
                .ins
                .push(String::from_utf8_lossy(bytes(b, &mut p)).into_owned()),
            2 => n
                .outs
                .push(String::from_utf8_lossy(bytes(b, &mut p)).into_owned()),
            3 => n.name = String::from_utf8_lossy(bytes(b, &mut p)).into_owned(),
            4 => n.op = String::from_utf8_lossy(bytes(b, &mut p)).into_owned(),
            5 => n.attrs.push(parse_attr(bytes(b, &mut p))),
            _ => skip(b, &mut p, wt),
        }
    }
    n
}
fn parse_value_info(b: &[u8]) -> String {
    let mut p = 0;
    let mut name = String::new();
    while p < b.len() {
        let (f, wt) = key(b, &mut p);
        match f {
            1 => name = String::from_utf8_lossy(bytes(b, &mut p)).into_owned(),
            _ => skip(b, &mut p, wt),
        }
    }
    name
}

pub fn parse(b: &[u8]) -> Graph {
    let mut p = 0;
    let mut g = Graph {
        nodes: Vec::new(),
        inits: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    while p < b.len() {
        let (f, wt) = key(b, &mut p);
        match f {
            7 => {
                let gb = bytes(b, &mut p);
                let mut q = 0;
                while q < gb.len() {
                    let (gf, gwt) = key(gb, &mut q);
                    match gf {
                        1 => g.nodes.push(parse_node(bytes(gb, &mut q))),
                        5 => {
                            let tb = bytes(gb, &mut q);
                            let t = parse_tensor(tb);
                            let mut r = 0;
                            let mut nm = String::new();
                            while r < tb.len() {
                                let (tf, twt) = key(tb, &mut r);
                                if tf == 8 {
                                    nm = String::from_utf8_lossy(bytes(tb, &mut r)).into_owned();
                                } else {
                                    skip(tb, &mut r, twt);
                                }
                            }
                            g.inits.push((nm, t));
                        }
                        11 => g.inputs.push(parse_value_info(bytes(gb, &mut q))),
                        12 => g.outputs.push(parse_value_info(bytes(gb, &mut q))),
                        _ => skip(gb, &mut q, gwt),
                    }
                }
            }
            _ => skip(b, &mut p, wt),
        }
    }
    g
}
