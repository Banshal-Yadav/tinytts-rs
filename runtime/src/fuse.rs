//! Graph-level pattern rewrites applied once at load.
//!
//! Four rewrites:
//!
//! 1. The `/head` sine-excitation generator, a 7-node chain over a `[1, 64, L]`
//!    tensor (5.26M elements at 54 tokens, 21 MB per intermediate). Fusing it
//!    removes 5 large intermediates and the per-element broadcast overhead.
//!    Verified against the golden tensors to 7.6e-6:
//!
//!    ```text
//!      f0    = Gather_7                       # pitch, [1, L]
//!      phase = Add_3                          # cumulative phase, [1, L]
//!      Mul_5 = (h+1) * f0                     # [1, 64, L]
//!      Less  = Mul_5 < 11000
//!      Cast  = f32(Less)
//!      Mul_6 = Cast * f32(f0 > 10)            # voiced gate
//!      Mul_7 = freq[h] * phase
//!      Sin   = sin(Mul_7)
//!      Mul_8 = Sin * Mul_6
//!      out   = ReduceSum(Mul_8, axis=1)       # [1, L]
//!    ```
//!
//! 2. `Transpose([0,2,3,1])` followed by 8 constant-index `Gather(axis=1)` nodes.
//!    The transpose materialises `[1,549,8,301] -> [1,8,301,549]` (1.32M
//!    elements) only for the gathers to slice it back apart. Replace the whole
//!    thing with 8 strided reads of the original tensor.
//!
//! 3. `DequantizeLinear` -> `Dequant`. The 105 dequant nodes produce 7.77M f32
//!    elements (31 MB) and the topological order puts every weight-only node
//!    first (steps 1034-1106) while their consumers run at 1663-2417, so 76 of
//!    them are live simultaneously: 28.9 MB, 62% of the live peak. `Dequant`
//!    keeps the int8 payload resident and materialises f32 on first use, then
//!    the exec loop drops the f32 right after the consumer runs.
//!
//! 4. `Mul`/`Add` chains over the same shape -> one `ElemChain` node, so the
//!    intermediate never reaches `vals` and the loop runs once instead of twice.

use crate::onnx::{Attr, Dt, Graph, Node};
use crate::tensor::T;

/// Apply every rewrite. Returns the number of nodes removed.
///
/// Set `FUSE=0` in the environment to disable all rewrites (used by the
/// differential test to check the unfused graph still matches ORT).
pub fn rewrite(g: &mut Graph) -> usize {
    if std::env::var("FUSE").map(|v| v == "0").unwrap_or(false) {
        return 0;
    }
    // `FUSE_SKIP=<name>` disables one pass, so its cost can be isolated by
    // difference.
    let skip = std::env::var("FUSE_SKIP").unwrap_or_default();
    let mut n = 0;
    if skip != "sine" {
        n += fuse_sine_excitation(g);
    }
    if skip != "transpose" {
        n += fuse_transpose_gather(g);
    }
    if skip != "dequant" {
        n += fuse_dequant(g);
    }
    if skip != "elemchain" {
        n += fuse_elem_chain(g);
    }
    n
}

/// Fuse `Mul`/`Add` chains into a single `ElemChain` node.
///
/// The graph has 392 `Mul`/`Add` nodes and 145 of them feed another `Mul`/`Add`
/// with no other consumer. Each one costs a dispatch, a `HashMap` lookup and a
/// fresh output buffer, and 112 of them operate on fewer than 1000 elements, so
/// the overhead dominates the arithmetic.
///
/// A chain `y = op1(a, b)` then `z = op2(y, c)` becomes one node with three
/// inputs and one output, evaluated in a single pass with no intermediate.
/// Only chains whose inner output has exactly one consumer are fused, and only
/// when the inner op is `Mul` or `Add` (both commutative, so operand order does
/// not matter for the rewrite).
pub fn fuse_elem_chain(g: &mut Graph) -> usize {
    // Build the consumer map once. The previous version rebuilt it from scratch
    // on every iteration of the fusion loop, i.e. 89 times over 2529 nodes, which
    // cost 48.6 ms of the 50 ms `Model::new` (15% of wall). Fusing one chain only
    // changes the map for the inner node's output and the outer node's inputs, so
    // it is maintained incrementally instead.
    let mut consumers: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (ni, n) in g.nodes.iter().enumerate() {
        for i in &n.ins {
            if !i.is_empty() {
                consumers.entry(i.clone()).or_default().push(ni);
            }
        }
    }

    let mut removed = 0usize;
    // A node is a fusion candidate only if it is a 2-input `Mul`/`Add` with one
    // output. Collect those once; the set only shrinks as chains are fused.
    let mut candidates: Vec<usize> = (0..g.nodes.len())
        .filter(|&ni| {
            let n = &g.nodes[ni];
            (n.op == "Mul" || n.op == "Add")
                && n.ins.len() == 2
                && n.outs.len() == 1
                && !n.outs[0].is_empty()
        })
        .collect();

    let mut qi = 0usize;
    while qi < candidates.len() {
        let inner = candidates[qi];
        qi += 1;
        // The inner node may have been consumed by an earlier fusion.
        if g.nodes[inner].op != "Mul" && g.nodes[inner].op != "Add" {
            continue;
        }
        let inner_out = g.nodes[inner].outs[0].clone();
        let cs = match consumers.get(&inner_out) {
            Some(c) => c,
            None => continue,
        };
        if cs.len() != 1 {
            continue;
        }
        let outer = cs[0];
        let c = &g.nodes[outer];
        if (c.op != "Mul" && c.op != "Add") || c.ins.len() != 2 || c.outs.len() != 1 {
            continue;
        }
        let (a, b) = (g.nodes[inner].ins[0].clone(), g.nodes[inner].ins[1].clone());
        let (ci0, ci1) = (g.nodes[outer].ins[0].clone(), g.nodes[outer].ins[1].clone());
        // The outer node's other operand is whichever input is not the inner out.
        let other = if ci0 == inner_out {
            ci1.clone()
        } else {
            ci0.clone()
        };
        let op1 = if g.nodes[inner].op == "Mul" { 2u8 } else { 0u8 };
        let op2 = if g.nodes[outer].op == "Mul" { 2u8 } else { 0u8 };
        let outs = g.nodes[outer].outs.clone();
        let name = g.nodes[outer].name.clone();
        g.nodes[outer] = Node {
            op: "ElemChain".into(),
            name,
            ins: vec![a, b, other],
            outs,
            attrs: vec![
                ("op1".into(), Attr::I(op1 as i64)),
                ("op2".into(), Attr::I(op2 as i64)),
            ],
        };
        // The inner node is now dead. Drop it from the consumer map and from the
        // outer node's input list, and re-queue the outer node: it may itself feed
        // another `Mul`/`Add` and so form a longer chain.
        consumers.remove(&inner_out);
        for i in [&ci0, &ci1] {
            if let Some(v) = consumers.get_mut(i) {
                v.retain(|&x| x != outer);
            }
        }
        g.nodes[inner].op = "Nop".into();
        g.nodes[inner].ins.clear();
        g.nodes[inner].outs.clear();
        removed += 1;
        candidates.push(outer);
    }
    removed
}

/// Rewrite every `DequantizeLinear` into a `Dequant` node that carries the int8
/// payload plus its scale/zero-point and materialises f32 lazily.
///
/// The node keeps the same inputs, so the graph topology is unchanged; only the
/// op name differs. `exec` resolves a `Dequant` output on demand and drops the
/// expanded f32 again once the consumer has run.
///
/// A `Dequant` whose only consumer is an `Identity` is folded into that Identity:
/// the Identity is a pure pass-through, but as a separate node it pins the
/// expanded f32 weight in `vals` from the Identity's step until the MatMul's
/// (30 of them, 0.786 MB each, live for ~600 steps).
pub fn fuse_dequant(g: &mut Graph) -> usize {
    let mut n = 0;
    for node in g.nodes.iter_mut() {
        if node.op == "DequantizeLinear" {
            node.op = "Dequant".into();
            n += 1;
        }
    }
    // Fold `Dequant -> Identity` into the Dequant, keeping the Identity's output
    // name so the consumer reads the expanded weight directly.
    //
    // A Dequant output may feed several Identities plus a MatMul (6 such cases),
    // so the fold is per-Identity: each Identity becomes a `DequantAlias` that
    // points at the same int8 payload. That keeps the expansion lazy for every
    // consumer instead of pinning one f32 copy per Identity.
    let mut fold: Vec<(usize, usize)> = Vec::new();
    for (di, d) in g.nodes.iter().enumerate() {
        if d.op != "Dequant" || d.outs.len() != 1 || d.outs[0].is_empty() {
            continue;
        }
        let out = d.outs[0].clone();
        for (ii, idn) in g.nodes.iter().enumerate() {
            if ii == di || idn.op != "Identity" || idn.outs.len() != 1 {
                continue;
            }
            if idn.ins.len() == 1 && idn.ins[0] == out {
                fold.push((di, ii));
            }
        }
    }
    let folded = fold.len();
    for (di, ii) in fold {
        // The alias needs the Dequant's own inputs (int8 payload, scale, zp), not
        // its output name, which is never stored in `vals`.
        let ins = g.nodes[di].ins.clone();
        g.nodes[ii].op = "DequantAlias".into();
        g.nodes[ii].ins = ins;
    }
    n + folded
}

/// `Transpose(perm=[0,2,3,1])` + N constant-index `Gather(axis=1)` -> N
/// `StridedSlice` nodes reading the pre-transpose tensor directly.
pub fn fuse_transpose_gather(g: &mut Graph) -> usize {
    // Find a Transpose with perm [0,2,3,1] whose consumers are all Gather(axis=1)
    // with a constant scalar index.
    let mut target: Option<(usize, Vec<(usize, i64)>)> = None;
    for (ti, n) in g.nodes.iter().enumerate() {
        if n.op != "Transpose" || n.outs.is_empty() {
            continue;
        }
        let perm = match n.attrs.iter().find(|(k, _)| k == "perm") {
            Some((_, Attr::Is(v))) => v.clone(),
            _ => continue,
        };
        if perm != vec![0, 2, 3, 1] {
            continue;
        }
        let out = n.outs[0].clone();
        let mut gathers = Vec::new();
        let mut ok = true;
        for (gi, c) in g.nodes.iter().enumerate() {
            if gi == ti || !c.ins.iter().any(|i| i == &out) {
                continue;
            }
            if c.op != "Gather" || c.ins.len() < 2 {
                ok = false;
                break;
            }
            let axis = c
                .attrs
                .iter()
                .find(|(k, _)| k == "axis")
                .map(|(_, a)| match a {
                    Attr::I(v) => *v,
                    _ => 0,
                });
            if axis != Some(1) {
                ok = false;
                break;
            }
            let Some(idx) = const_scalar_i(g, &c.ins[1]) else {
                ok = false;
                break;
            };
            gathers.push((gi, idx));
        }
        if ok && !gathers.is_empty() {
            target = Some((ti, gathers));
            break;
        }
    }
    let Some((ti, gathers)) = target else {
        return 0;
    };

    let src = g.nodes[ti].ins[0].clone();
    let mut dead: Vec<usize> = vec![ti];
    for (gi, idx) in &gathers {
        let outs = g.nodes[*gi].outs.clone();
        g.nodes[*gi] = Node {
            op: "StridedSlice".into(),
            name: g.nodes[*gi].name.clone(),
            ins: vec![src.clone()],
            outs,
            attrs: vec![("index".into(), Attr::I(*idx))],
        };
        dead.push(*gi);
    }
    // The Transpose is now unused; drop it.
    dead.sort_unstable();
    dead.dedup();
    let removed = dead.len();
    for &i in dead.iter().rev() {
        if g.nodes[i].op == "Transpose" {
            g.nodes.remove(i);
        }
    }
    removed - 1
}

/// The fused elementwise chain kernel.
///
/// `out = op2(op1(a, b), c)` where `op1`/`op2` are 0 (`Add`) or 2 (`Mul`).
/// Evaluated in one pass with no intermediate buffer. Broadcasting follows the
/// same rules as `elem::binary`: same-shape, scalar, last-dim, per-channel and
/// outer-product are the shapes the graph actually uses.
pub fn elem_chain(a: &T, b: &T, c: &T, op1: u8, op2: u8) -> T {
    // Only f32 chains are fused; anything else falls back to the two separate
    // ops so the dtype rules stay in one place (`elem::binary`).
    if a.dt != Dt::F32 || b.dt != Dt::F32 || c.dt != Dt::F32 {
        let t = crate::kernels::elem::binary(a, b, op1);
        return crate::kernels::elem::binary(&t, c, op2);
    }
    let dims = broadcast_dims(&[a, b, c]);
    let n: usize = dims.iter().product();
    let mut out = T::new(Dt::F32, dims.clone());
    let ov = out.fm();
    let av = a.f();
    let bv = b.f();
    let cv = c.f();
    // Fast path: all three are the same shape (or scalars), which is the common
    // case in the `/head` chain. The op pair is hoisted out of the loop so the
    // inner body is a straight-line FMA.
    if a.len() == n && b.len() == n && c.len() == n {
        match (op1, op2) {
            (2, 2) => {
                for i in 0..n {
                    ov[i] = av[i] * bv[i] * cv[i];
                }
            }
            (2, 0) => {
                for i in 0..n {
                    ov[i] = av[i] * bv[i] + cv[i];
                }
            }
            (0, 2) => {
                for i in 0..n {
                    ov[i] = (av[i] + bv[i]) * cv[i];
                }
            }
            (0, 0) => {
                for i in 0..n {
                    ov[i] = av[i] + bv[i] + cv[i];
                }
            }
            _ => {
                for i in 0..n {
                    let t = if op1 == 2 {
                        av[i] * bv[i]
                    } else {
                        av[i] + bv[i]
                    };
                    ov[i] = if op2 == 2 { t * cv[i] } else { t + cv[i] };
                }
            }
        }
        return out;
    }
    // Fast path: exactly one operand is a scalar and the other two are at the
    // output shape.
    let scalars = [a.len() == 1, b.len() == 1, c.len() == 1];
    if scalars.iter().filter(|&&s| s).count() == 1 {
        let (sa, sb, sc) = (av[0], bv[0], cv[0]);
        let (xa, xb, xc) = (a.len() == n, b.len() == n, c.len() == n);
        if (xa as usize + xb as usize + xc as usize) == 2 {
            for i in 0..n {
                let va = if xa { av[i] } else { sa };
                let vb = if xb { bv[i] } else { sb };
                let vc = if xc { cv[i] } else { sc };
                let t = if op1 == 2 { va * vb } else { va + vb };
                ov[i] = if op2 == 2 { t * vc } else { t + vc };
            }
            return out;
        }
    }
    // Fast path: two operands share the output shape and the third is a
    // per-channel `[1, C, 1]` or last-dim `[.., 1]` vector.
    if let Some(()) = elem_chain_bcast(&[a, b, c], &dims, ov, op1, op2) {
        return out;
    }
    // General path: index each operand with its own broadcast strides.
    let sa = bcast_strides(&a.dims, &dims);
    let sb = bcast_strides(&b.dims, &dims);
    let sc = bcast_strides(&c.dims, &dims);
    let ost = crate::tensor::strides(&dims);
    let mut idx = vec![0usize; dims.len()];
    for i in 0..n {
        let mut rem = i;
        for d in 0..dims.len() {
            idx[d] = rem / ost[d];
            rem %= ost[d];
        }
        let ia: usize = (0..dims.len()).map(|d| idx[d] * sa[d]).sum();
        let ib: usize = (0..dims.len()).map(|d| idx[d] * sb[d]).sum();
        let ic: usize = (0..dims.len()).map(|d| idx[d] * sc[d]).sum();
        let t = if op1 == 2 {
            av[ia] * bv[ib]
        } else {
            av[ia] + bv[ib]
        };
        ov[i] = if op2 == 2 { t * cv[ic] } else { t + cv[ic] };
    }
    out
}

/// Fast path for the broadcast shapes the graph uses: two operands at the
/// output shape and the third either `[1, C, 1]` (per-channel) or `[.., 1]`
/// (last-dim). Returns `None` when the shapes need the general path.
fn elem_chain_bcast(ts: &[&T; 3], dims: &[usize], ov: &mut [f32], op1: u8, op2: u8) -> Option<()> {
    let n: usize = dims.iter().product();
    // Which operand is the odd one out?
    let full: Vec<usize> = (0..3).filter(|&i| ts[i].len() == n).collect();
    if full.len() != 2 {
        return None;
    }
    let odd = (0..3).find(|i| !full.contains(i))?;
    let o = ts[odd];
    // Per-channel: [1, C, 1] against [1, C, L].
    let per_channel = dims.len() == 3
        && o.dims.len() == 3
        && o.dims[0] == 1
        && o.dims[2] == 1
        && o.dims[1] == dims[1];
    // Last-dim: [.., 1] against [.., L].
    let last_dim = o.dims.len() == dims.len()
        && o.dims[..o.dims.len() - 1] == dims[..dims.len() - 1]
        && *o.dims.last().unwrap_or(&0) == 1;
    if !per_channel && !last_dim {
        return None;
    }
    let ov_v = o.f();
    let (i0, i1) = (full[0], full[1]);
    let (v0, v1) = (ts[i0].f(), ts[i1].f());
    let inner = if per_channel { dims[2] } else { 1 };
    let outer = n / inner.max(1);
    // The odd operand must have exactly one value per outer index.
    if ov_v.len() != outer {
        return None;
    }
    for oc in 0..outer {
        let s = ov_v[oc];
        let base = oc * inner;
        for j in 0..inner {
            let i = base + j;
            let (va, vb, vc) = match odd {
                0 => (s, v0[i], v1[i]),
                1 => (v0[i], s, v1[i]),
                _ => (v0[i], v1[i], s),
            };
            let t = if op1 == 2 { va * vb } else { va + vb };
            ov[i] = if op2 == 2 { t * vc } else { t + vc };
        }
    }
    Some(())
}

/// Broadcast the three operand shapes to a common shape.
fn broadcast_dims(ts: &[&T]) -> Vec<usize> {
    let r = ts.iter().map(|t| t.dims.len()).max().unwrap_or(0);
    let mut out = vec![1usize; r];
    for t in ts {
        let off = r - t.dims.len();
        for (i, &d) in t.dims.iter().enumerate() {
            out[off + i] = out[off + i].max(d);
        }
    }
    out
}

/// Strides that map an index in `dims` to an index in `src`, with 0 for
/// broadcast axes.
fn bcast_strides(src: &[usize], dims: &[usize]) -> Vec<usize> {
    let r = dims.len();
    let off = r - src.len();
    let mut st = vec![0usize; r];
    let mut acc = 1usize;
    for i in (0..src.len()).rev() {
        if src[i] != 1 {
            st[off + i] = acc;
        }
        acc *= src[i];
    }
    st
}

/// `out[c, p] = x[p, index, c]` for a `[1, P, J, C]` input.
pub fn strided_slice(x: &T, index: i64) -> T {
    let (p, j, c) = (x.dims[1], x.dims[2], x.dims[3]);
    let jj = index as usize;
    let mut out = T::new(Dt::F32, vec![1, c, p]);
    let xv = x.f();
    let ov = out.fm();
    for pi in 0..p {
        let base = pi * j * c + jj * c;
        for ci in 0..c {
            ov[ci * p + pi] = xv[base + ci];
        }
    }
    out
}

fn const_scalar_i(g: &Graph, name: &str) -> Option<i64> {
    if let Some((_, t)) = g.inits.iter().find(|(n, _)| n == name) {
        return match t.dt {
            Dt::I64 => Some(i64::from_le_bytes(t.data[..8].try_into().unwrap())),
            Dt::I32 => Some(i32::from_le_bytes(t.data[..4].try_into().unwrap()) as i64),
            _ => None,
        };
    }
    let n = g.nodes.iter().find(|n| n.outs.iter().any(|o| o == name))?;
    for (k, a) in &n.attrs {
        if k == "value" {
            match a {
                Attr::I(v) => return Some(*v),
                Attr::T(b) | Attr::S(b) => {
                    let t = crate::onnx::parse_tensor(b);
                    return match t.dt {
                        Dt::I64 => Some(i64::from_le_bytes(t.data[..8].try_into().unwrap())),
                        Dt::I32 => Some(i32::from_le_bytes(t.data[..4].try_into().unwrap()) as i64),
                        _ => None,
                    };
                }
                _ => {}
            }
        }
    }
    None
}

/// Rewrite the sine-excitation chain into a single `SineExcitation` node.
/// Returns the number of nodes removed.
pub fn fuse_sine_excitation(g: &mut Graph) -> usize {
    // Locate the ReduceSum that consumes a Mul whose inputs are a Sin and a Mul.
    let mut target: Option<usize> = None;
    for (i, n) in g.nodes.iter().enumerate() {
        if n.op != "ReduceSum" || n.ins.len() < 2 {
            continue;
        }
        let Some(mul8) = producer_of(g, &n.ins[0]) else {
            continue;
        };
        if g.nodes[mul8].op != "Mul" || g.nodes[mul8].ins.len() != 2 {
            continue;
        }
        let (a, b) = (g.nodes[mul8].ins[0].clone(), g.nodes[mul8].ins[1].clone());
        let (sin_i, gate_i) = match (producer_of(g, &a), producer_of(g, &b)) {
            (Some(s), Some(t)) if g.nodes[s].op == "Sin" => (s, t),
            (Some(s), Some(t)) if g.nodes[t].op == "Sin" => (t, s),
            _ => continue,
        };
        // gate = Mul(Cast(Less(Mul(f0, k), thr)), Unsqueeze(Greater(f0, gthr)))
        let Some(cast_i) = producer_of(g, &g.nodes[gate_i].ins[0]) else {
            continue;
        };
        if g.nodes[cast_i].op != "Cast" {
            continue;
        }
        let Some(less_i) = producer_of(g, &g.nodes[cast_i].ins[0]) else {
            continue;
        };
        if g.nodes[less_i].op != "Less" {
            continue;
        }
        let Some(mul5_i) = producer_of(g, &g.nodes[less_i].ins[0]) else {
            continue;
        };
        if g.nodes[mul5_i].op != "Mul" {
            continue;
        }
        // Mul_7 = freq * Unsqueeze(phase)
        let Some(mul7_i) = producer_of(g, &g.nodes[sin_i].ins[0]) else {
            continue;
        };
        if g.nodes[mul7_i].op != "Mul" {
            continue;
        }
        target = Some(i);
        break;
    }
    let Some(rs_i) = target else { return 0 };

    // Resolve the operands.
    let rs = g.nodes[rs_i].clone();
    let mul8 = g.nodes[producer_of(g, &rs.ins[0]).unwrap()].clone();
    let (sin_i, gate_i) = {
        let (a, b) = (mul8.ins[0].clone(), mul8.ins[1].clone());
        match (producer_of(g, &a), producer_of(g, &b)) {
            (Some(s), Some(t)) if g.nodes[s].op == "Sin" => (s, t),
            (Some(s), Some(t)) if g.nodes[t].op == "Sin" => (t, s),
            _ => return 0,
        }
    };
    let sin = g.nodes[sin_i].clone();
    let gate = g.nodes[gate_i].clone();
    let cast = g.nodes[producer_of(g, &gate.ins[0]).unwrap()].clone();
    let less = g.nodes[producer_of(g, &cast.ins[0]).unwrap()].clone();
    let mul5 = g.nodes[producer_of(g, &less.ins[0]).unwrap()].clone();
    let mul7 = g.nodes[producer_of(g, &sin.ins[0]).unwrap()].clone();

    // f0: the input of Mul_5 that is not the harmonic-index vector `head.k`.
    // `head.k` is a Cast of an fp16 initializer of shape [1,64,1].
    let f0_name = mul5
        .ins
        .iter()
        .find(|s| !is_harmonic_index(g, s))
        .cloned()
        .unwrap_or_default();
    // phase: the input of Mul_7 that is not the frequency vector.
    let phase_name = mul7
        .ins
        .iter()
        .find(|s| !is_freq_vector(g, s))
        .cloned()
        .unwrap_or_default();
    if f0_name.is_empty() || phase_name.is_empty() {
        return 0;
    }
    // freq vector: the other input of Mul_7.
    let freq_name = mul7
        .ins
        .iter()
        .find(|s| is_freq_vector(g, s))
        .cloned()
        .unwrap_or_default();
    // thresholds
    let thr = if less.ins.len() > 1 {
        const_scalar(g, &less.ins[1]).unwrap_or(11000.0)
    } else {
        11000.0
    };
    let gthr = {
        // gate.ins[1] is Unsqueeze(Greater(f0, gthr)); unwrap to the Greater node.
        let mut cur = producer_of(g, &gate.ins[1]);
        for _ in 0..3 {
            let Some(i) = cur else { break };
            let n = &g.nodes[i];
            if n.op == "Greater" {
                cur = Some(i);
                break;
            }
            if n.ins.is_empty() {
                cur = None;
                break;
            }
            cur = producer_of(g, &n.ins[0]);
        }
        cur.and_then(|i| {
            let n = &g.nodes[i];
            if n.op == "Greater" && n.ins.len() > 1 {
                const_scalar(g, &n.ins[1])
            } else {
                None
            }
        })
        .unwrap_or(10.0)
    };

    // Build the fused node.
    let mut fused = Node {
        op: "SineExcitation".into(),
        name: "/head/SineExcitation".into(),
        ins: vec![f0_name, phase_name, freq_name],
        outs: rs.outs.clone(),
        attrs: vec![("thr".into(), Attr::F(thr)), ("gthr".into(), Attr::F(gthr))],
    };
    fused.ins.retain(|s| !s.is_empty());

    // Remove the 8 chain nodes and append the fused one.
    let mut dead: Vec<usize> = vec![
        rs_i,
        producer_of(g, &rs.ins[0]).unwrap(),
        sin_i,
        gate_i,
        producer_of(g, &gate.ins[0]).unwrap(),
        producer_of(g, &cast.ins[0]).unwrap(),
        producer_of(g, &less.ins[0]).unwrap(),
        producer_of(g, &sin.ins[0]).unwrap(),
    ];
    dead.sort_unstable();
    dead.dedup();
    let removed = dead.len();
    for &i in dead.iter().rev() {
        g.nodes.remove(i);
    }
    g.nodes.push(fused);
    removed - 1
}

fn producer_of(g: &Graph, name: &str) -> Option<usize> {
    g.nodes
        .iter()
        .position(|n| n.outs.iter().any(|o| o == name))
}

/// `head.k` is `Cast(fp16 init [1,64,1])` — the harmonic index vector.
fn is_harmonic_index(g: &Graph, name: &str) -> bool {
    cast_src_is(g, name, "head.k_16")
}

/// The frequency vector is `Cast(fp16 init [1,64,1])` named `onnx::Mul_4194_16`.
fn is_freq_vector(g: &Graph, name: &str) -> bool {
    cast_src_is(g, name, "onnx::Mul_4194_16")
}

fn cast_src_is(g: &Graph, name: &str, init: &str) -> bool {
    let Some(i) = producer_of(g, name) else {
        return false;
    };
    let n = &g.nodes[i];
    n.op == "Cast" && n.ins.first().map(|s| s == init).unwrap_or(false)
}

fn const_scalar(g: &Graph, name: &str) -> Option<f32> {
    if let Some((_, t)) = g.inits.iter().find(|(n, _)| n == name) {
        return match t.dt {
            Dt::F32 => Some(f32::from_le_bytes(t.data[..4].try_into().unwrap())),
            Dt::F64 => Some(f64::from_le_bytes(t.data[..8].try_into().unwrap()) as f32),
            Dt::F16 => Some(crate::quant::f16_to_f32(u16::from_le_bytes(
                t.data[..2].try_into().unwrap(),
            ))),
            _ => None,
        };
    }
    let n = g.nodes.iter().find(|n| n.outs.iter().any(|o| o == name))?;
    for (k, a) in &n.attrs {
        if k == "value" {
            match a {
                Attr::F(v) => return Some(*v),
                Attr::T(b) | Attr::S(b) => {
                    let t = crate::onnx::parse_tensor(b);
                    if t.dt == Dt::F32 {
                        return Some(f32::from_le_bytes(t.data[..4].try_into().unwrap()));
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// The fused kernel.
///
/// `out[t] = gate[t] * sum_h sin(freq[h] * phase[t]) * f32((h+1)*f0[t] < thr)`
/// where `gate[t] = f32(f0[t] > gthr)`.
///
/// The harmonic cutoff `(h+1)*f0[t] < thr` is monotone in `h`, so it is solved
/// once per `t` for the largest included `h` instead of being tested 64 times.
pub fn sine_excitation(f0: &T, phase: &T, freq: &T, thr: f32, gthr: f32) -> T {
    let l = f0.len();
    let h = freq.len();
    let f0v = f0.f();
    let phv = phase.f();
    let frv = freq.f();
    let mut out = T::new(Dt::F32, vec![1, l]);
    let ov = out.fm();
    for t in 0..l {
        let f0t = f0v[t];
        if f0t <= gthr {
            ov[t] = 0.0;
            continue;
        }
        // Largest h with (h+1)*f0t < thr, i.e. h < thr/f0t - 1. The bound is
        // strict, so the count is ceil(thr/f0t - 1) clamped to [0, h].
        let hmax = if f0t > 0.0 {
            let lim = thr / f0t - 1.0;
            if lim <= 0.0 {
                0
            } else if lim >= h as f32 {
                h
            } else {
                let c = lim.ceil() as usize;
                c.min(h)
            }
        } else {
            h
        };
        let pht = phv[t];
        let mut acc = 0f32;
        for hi in 0..hmax {
            acc += (frv[hi] * pht).sin();
        }
        ov[t] = acc;
    }
    out
}
