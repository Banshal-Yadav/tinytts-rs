use crate::onnx::Dt;
use crate::tensor::T;

pub fn matmul(a: &T, b: &T) -> T {
    let ar = a.dims.len();
    let br = b.dims.len();
    let (m, k) = (a.dims[ar - 2], a.dims[ar - 1]);
    let (k2, n) = (b.dims[br - 2], b.dims[br - 1]);
    assert_eq!(k, k2, "matmul k mismatch {k} {k2}");
    let batch: usize = a.dims[..ar - 2]
        .iter()
        .product::<usize>()
        .max(b.dims[..br - 2].iter().product());
    let mut od: Vec<usize> = if ar >= br {
        a.dims[..ar - 2].to_vec()
    } else {
        b.dims[..br - 2].to_vec()
    };
    od.push(m);
    od.push(n);
    let mut out = T::new(Dt::F32, od);
    let av = a.f();
    let bv = b.f();
    let o = out.fm();
    let abatch = a.dims[..ar - 2].iter().product::<usize>().max(1);
    let bbatch = b.dims[..br - 2].iter().product::<usize>().max(1);
    for bi in 0..batch {
        let ao = if abatch == 1 { 0 } else { bi * m * k };
        let bo = if bbatch == 1 { 0 } else { bi * k * n };
        gemm(
            m,
            n,
            k,
            &av[ao..ao + m * k],
            &bv[bo..bo + k * n],
            &mut o[bi * m * n..(bi + 1) * m * n],
            None,
            None,
        );
    }
    out
}

#[inline]
pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    bias: Option<&[f32]>,
    act: Option<u8>,
) {
    crate::kernels::sgemm::gemm(m, n, k, a, b, c);
    if let Some(bs) = bias {
        for i in 0..m {
            let row = &mut c[i * n..i * n + n];
            for (v, &bv) in row.iter_mut().zip(bs.iter()) {
                *v += bv;
            }
        }
    }
    if let Some(a) = act {
        for v in c[..m * n].iter_mut() {
            *v = match a {
                1 => v.max(0.0),
                2 => crate::quant::sigmoid(*v),
                3 => v.tanh(),
                4 => {
                    let x = *v;
                    x / (1.0 + (-x).exp())
                }
                _ => *v,
            };
        }
    }
}

pub fn gemm_nt(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
    bias: Option<&[f32]>,
) {
    for i in 0..m {
        let arow = &a[i * k..i * k + k];
        let crow = &mut c[i * n..i * n + n];
        match bias {
            Some(bs) => crow.copy_from_slice(&bs[..n]),
            None => crow.fill(0.0),
        }
        for j in 0..n {
            let brow = &b_t[j * k..j * k + k];
            let mut s = 0.0f32;
            for kk in 0..k {
                s += arow[kk] * brow[kk];
            }
            crow[j] += s;
        }
    }
}
