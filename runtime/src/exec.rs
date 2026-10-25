use crate::graph::Model;
use crate::kernels::{conv, elem, matmul, nn, shape};
use crate::onnx::{Attr, Dt, Node};
use crate::quant::{dequant_f16, dequant_i8_f32, f16_to_f32, Rng};
use crate::tensor::{norm_axis, T};
use std::collections::HashMap;
use std::sync::Arc;

pub struct Exec {
    pub vals: HashMap<String, T>,
    pub rng: Rng,
    pub lstm_cache: HashMap<String, nn::LstmW>,
    /// `Dequant` node output name -> (int8 payload, scale, zero-point, per_row).
    pub dequant_src: HashMap<String, DequantSrc>,
    /// When true, `run` times every node and prints a breakdown.
    pub profile: bool,
    /// When true, drop each tensor from `vals` once its last consumer has run.
    /// Cuts peak RSS sharply; off by default so the differential test can still
    /// teacher-force arbitrary tensors.
    pub free_dead: bool,
    /// When true, accumulate coarse phase timings for the whole run instead of
    /// per-node timings. Per-node `Instant::now()` costs ~7 us per node and
    /// inflates the sum above the real synth time; this mode uses one timer per
    /// phase so the numbers add up to the measured wall.
    pub account: bool,
    /// (kernel_ns, dispatch_ns, dequant_ns) accumulated when `account` is set.
    pub acct: [u64; 3],
    /// Per-op-type (bytes allocated, allocation count) when `account` is set.
    pub acct_alloc: HashMap<String, (u64, u64)>,
    /// Per-op-type kernel nanoseconds when `account` is set. Only ops in
    /// `acct_expensive` are timed individually, so the timer overhead lands on
    /// the ops that can absorb it instead of being spread over 2529 nodes.
    pub acct_op: HashMap<String, u64>,
    /// Op types to time individually. Computed by `run` from a cheap first pass
    /// that counts nodes per op; ops with few nodes are timed, ops with many
    /// tiny nodes are not.
    pub acct_expensive: std::collections::HashSet<String>,
    /// Minimum average node cost (ns) for an op to be timed individually.
    pub acct_min_ns: u64,
    /// When set, time *every* node individually regardless of count. This is the
    /// `--profile` behaviour and carries the same ~25 ms of timer overhead, but
    /// it is the only way to get a per-op number for the many-tiny-node ops.
    /// The harness runs both modes and reports the difference.
    pub acct_all: bool,
    /// When set, record per-node timings (name -> ns) in `acct_node`.
    pub acct_nodes: bool,
    /// Per-node kernel nanoseconds, keyed by node name. Only populated when
    /// `acct_nodes` is set.
    pub acct_node: HashMap<String, u64>,
}

/// The int8 payload of a `Dequant` node plus everything needed to expand it.
pub struct DequantSrc {
    pub q: T,
    pub scale: Vec<f32>,
    pub zp: Vec<i8>,
    pub per_row: usize,
}

impl Default for Exec {
    fn default() -> Exec {
        Exec::new()
    }
}

impl Exec {
    pub fn new() -> Exec {
        Exec {
            vals: HashMap::new(),
            rng: Rng::new(0x9E3779B97F4A7C15),
            lstm_cache: HashMap::new(),
            dequant_src: HashMap::new(),
            profile: false,
            free_dead: false,
            account: false,
            acct: [0; 3],
            acct_alloc: HashMap::new(),
            acct_op: HashMap::new(),
            acct_expensive: std::collections::HashSet::new(),
            acct_min_ns: 0,
            acct_all: false,
            acct_nodes: false,
            acct_node: HashMap::new(),
        }
    }

    /// Run with liveness-based freeing enabled.
    pub fn run_freeing(&mut self, m: &Model, feeds: Vec<(String, T)>) -> Vec<T> {
        self.free_dead = true;
        let r = self.run(m, feeds);
        self.free_dead = false;
        r
    }

    pub fn run(&mut self, m: &Model, feeds: Vec<(String, T)>) -> Vec<T> {
        self.vals.clear();
        for (n, t) in &m.g.inits {
            self.vals.insert(
                n.clone(),
                T {
                    dt: t.dt,
                    dims: t.dims.clone(),
                    d: t.data.clone(),
                },
            );
        }
        for (n, t) in feeds {
            self.vals.insert(n, t);
        }
        let profile = self.profile;
        let free_dead = self.free_dead;
        // Last consumer index per tensor name, for liveness-based freeing.
        let last_use: HashMap<&str, usize> = if free_dead {
            let mut lu: HashMap<&str, usize> = HashMap::new();
            for (step, &ni) in m.order.iter().enumerate() {
                for i in &m.g.nodes[ni].ins {
                    if !i.is_empty() {
                        lu.insert(i.as_str(), step);
                    }
                }
            }
            lu
        } else {
            HashMap::new()
        };
        let outputs: std::collections::HashSet<&str> =
            m.g.outputs.iter().map(|s| s.as_str()).collect();
        let mut by_op: HashMap<String, (f64, usize)> = HashMap::new();
        let mut by_node: Vec<(f64, String, String)> = Vec::new();
        let account = self.account;
        // Cheap first pass: count nodes per op type. An op is timed individually
        // only if it has few enough nodes that the timer overhead is negligible
        // relative to its work. Ops with hundreds of tiny nodes (Cast, Add,
        // Transpose, Slice) are excluded, because timing them would add ~7 us per
        // node and swamp their real cost.
        //
        // The threshold is on node count, but ops excluded here are still
        // reported: their combined cost is the difference between the kernel
        // total and the sum of the timed ops, and the harness prints it as
        // "untimed remainder" so they are not silently shown as zero.
        if account && self.acct_min_ns > 0 {
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for &ni in m.order.iter() {
                *counts.entry(m.g.nodes[ni].op.as_str()).or_insert(0) += 1;
            }
            self.acct_expensive = counts
                .into_iter()
                .filter(|(_, c)| *c <= 100)
                .map(|(op, _)| op.to_string())
                .collect();
        }
        for (step, &ni) in m.order.iter().enumerate() {
            let node = &m.g.nodes[ni];
            let t0 = if profile {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let ta = if account {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let a0 = if account { crate::alloc::allocs() } else { 0 };
            let b0 = if account { crate::alloc::bytes() } else { 0 };
            // Time this node individually only if its op type is in the expensive
            // set, which `run` computes in a cheap first pass. Cheap ops never pay
            // the timer cost, so the overhead lands only on ops that can absorb it.
            let t1 = if account
                && (self.acct_all || self.acct_nodes || self.acct_expensive.contains(&node.op))
            {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let outs = self.eval(node);
            if let Some(t1) = t1 {
                let ns = t1.elapsed().as_nanos() as u64;
                *self.acct_op.entry(node.op.clone()).or_insert(0) += ns;
                if self.acct_nodes {
                    // Key by the first output name: many nodes have an empty
                    // `name`, but every node has at least one output.
                    let key = node
                        .outs
                        .iter()
                        .find(|o| !o.is_empty())
                        .cloned()
                        .unwrap_or_else(|| format!("{}#{}", node.op, step));
                    *self.acct_node.entry(key).or_insert(0) += ns;
                }
            }
            if let Some(ta) = ta {
                self.acct[0] += ta.elapsed().as_nanos() as u64;
                let e = self.acct_alloc.entry(node.op.clone()).or_insert((0, 0));
                e.0 += crate::alloc::bytes() - b0;
                e.1 += (crate::alloc::allocs() - a0) as u64;
            }
            if let Some(t0) = t0 {
                let dt = t0.elapsed().as_secs_f64();
                let e = by_op.entry(node.op.clone()).or_insert((0.0, 0));
                e.0 += dt;
                e.1 += 1;
                by_node.push((dt, node.op.clone(), node.name.clone()));
            }
            let tb = if account {
                Some(std::time::Instant::now())
            } else {
                None
            };
            for (i, o) in node.outs.iter().enumerate() {
                if o.is_empty() {
                    continue;
                }
                if i < outs.len() {
                    self.vals.insert(o.clone(), outs[i].clone());
                }
            }
            if free_dead {
                for i in &node.ins {
                    if i.is_empty() || outputs.contains(i.as_str()) {
                        continue;
                    }
                    if last_use.get(i.as_str()) == Some(&step) {
                        self.vals.remove(i);
                    }
                }
            }
            if let Some(tb) = tb {
                self.acct[1] += tb.elapsed().as_nanos() as u64;
            }
        }
        if profile {
            self.report_profile(&by_op, &mut by_node);
        }
        m.g.outputs
            .iter()
            .map(|o| self.vals.remove(o).expect("missing output"))
            .collect()
    }

    fn report_profile(
        &self,
        by_op: &HashMap<String, (f64, usize)>,
        by_node: &mut [(f64, String, String)],
    ) {
        let total: f64 = by_op.values().map(|(t, _)| t).sum();
        let mut ops: Vec<_> = by_op.iter().collect();
        ops.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());

        eprintln!();
        eprintln!("=== per-op-type (total {:.3} s) ===", total);
        eprintln!("{:<24} {:>10} {:>8} {:>7}", "op", "ms", "%", "n");
        for (op, (t, n)) in &ops {
            eprintln!(
                "{op:<24} {:>10.1} {:>7.1}% {:>7}",
                t * 1e3,
                t / total * 100.0,
                n
            );
        }

        by_node.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        eprintln!();
        eprintln!("=== top 15 individual nodes ===");
        eprintln!("{:<24} {:>10} {:>7}  name", "op", "ms", "%");
        for (t, op, name) in by_node.iter().take(15) {
            let nm = if name.is_empty() { "-" } else { name.as_str() };
            let nm = if nm.len() > 46 { &nm[..46] } else { nm };
            eprintln!(
                "{op:<24} {:>10.2} {:>6.1}%  {}",
                t * 1e3,
                t / total * 100.0,
                nm
            );
        }
        eprintln!();
    }

    /// Load initializers + feeds into `vals` without executing anything.
    pub fn load(&mut self, m: &Model, feeds: Vec<(String, T)>) {
        self.vals.clear();
        for (n, t) in &m.g.inits {
            self.vals.insert(
                n.clone(),
                T {
                    dt: t.dt,
                    dims: t.dims.clone(),
                    d: t.data.clone(),
                },
            );
        }
        for (n, t) in feeds {
            self.vals.insert(n, t);
        }
    }

    /// Execute a single node against whatever is currently in `vals`.
    /// Used by the differential test to teacher-force ORT's real inputs.
    pub fn eval_node(&mut self, n: &Node) -> Vec<T> {
        self.eval(n)
    }

    /// Run the graph and print one line per node: `op<TAB>name<TAB>in_shapes<TAB>out_shapes`.
    /// Used by `scripts/bench/gflops.py` to compute exact MAC counts.
    pub fn run_dump_shapes(&mut self, m: &Model, feeds: Vec<(String, T)>) {
        self.vals.clear();
        for (n, t) in &m.g.inits {
            self.vals.insert(
                n.clone(),
                T {
                    dt: t.dt,
                    dims: t.dims.clone(),
                    d: t.data.clone(),
                },
            );
        }
        for (n, t) in feeds {
            self.vals.insert(n, t);
        }
        for &ni in m.order.iter() {
            let node = &m.g.nodes[ni];
            // `Dequant` outputs are expanded on demand; materialise them so the
            // weight shapes are visible.
            for i in &node.ins {
                if !i.is_empty() {
                    self.materialize(i);
                }
            }
            let ins: Vec<String> = node
                .ins
                .iter()
                .map(|i| {
                    if i.is_empty() {
                        "-".to_string()
                    } else {
                        match self.vals.get(i) {
                            Some(t) => t
                                .dims
                                .iter()
                                .map(|d| d.to_string())
                                .collect::<Vec<_>>()
                                .join("x"),
                            None => "?".to_string(),
                        }
                    }
                })
                .collect();
            let outs = self.eval(node);
            let os: Vec<String> = outs
                .iter()
                .map(|t| {
                    t.dims
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("x")
                })
                .collect();
            println!(
                "{}\t{}\t{}\t{}",
                node.op,
                node.name,
                ins.join(","),
                os.join(",")
            );
            for (i, o) in node.outs.iter().enumerate() {
                if o.is_empty() || i >= outs.len() {
                    continue;
                }
                self.vals.insert(o.clone(), outs[i].clone());
            }
        }
    }

    /// Insert a tensor into `vals` (teacher-forcing).
    pub fn put(&mut self, name: &str, t: T) {
        self.vals.insert(name.to_string(), t);
    }

    /// Read a tensor from `vals`.
    pub fn peek(&self, name: &str) -> Option<&T> {
        self.vals.get(name)
    }

    /// Materialise a `Dequant` output into `vals` if it is not already there.
    /// Returns true when the name is now readable via `peek`.
    ///
    /// The differential test needs this: `Dequant` outputs are never stored in
    /// `vals` during a normal run (they are expanded on demand and dropped again),
    /// so a teacher-forcing harness that inspects `peek` would see them missing.
    pub fn materialize(&mut self, name: &str) -> bool {
        if self.vals.contains_key(name) {
            return true;
        }
        let Some(src) = self.dequant_src.get(name) else {
            return false;
        };
        let t = expand_dequant(src);
        self.vals.insert(name.to_string(), t);
        true
    }

    #[inline]
    fn get(&self, n: &str) -> &T {
        if let Some(t) = self.vals.get(n) {
            return t;
        }
        if let Some(t) = self.vals.get(&format!("{n}16")) {
            return t;
        }
        panic!("missing tensor {n}")
    }

    fn opt(&self, n: &str) -> Option<&T> {
        if n.is_empty() {
            None
        } else {
            self.vals
                .get(n)
                .or_else(|| self.vals.get(&format!("{n}16")))
        }
    }

    fn eval(&mut self, n: &Node) -> Vec<T> {
        // Materialise any `Dequant` inputs this node needs, then drop them again
        // once the node has run. Only the weights one node actually uses are ever
        // expanded, so the 31 MB of f32 weights is never all live at once.
        let mut materialised: Vec<String> = Vec::new();
        let account = self.account;
        let td = if account {
            Some(std::time::Instant::now())
        } else {
            None
        };
        for i in &n.ins {
            if i.is_empty() || self.vals.contains_key(i) {
                continue;
            }
            if let Some(src) = self.dequant_src.get(i) {
                let t = expand_dequant(src);
                self.vals.insert(i.clone(), t);
                materialised.push(i.clone());
            }
        }
        if let Some(td) = td {
            self.acct[2] += td.elapsed().as_nanos() as u64;
        }
        let out = self.eval_inner(n);
        for name in materialised {
            self.vals.remove(&name);
        }
        out
    }

    fn eval_inner(&mut self, n: &Node) -> Vec<T> {
        let g = |i: usize| -> &T {
            self.vals
                .get(&n.ins[i])
                .unwrap_or_else(|| panic!("missing tensor {}", n.ins[i]))
        };
        let o = |i: usize| {
            if i < n.ins.len() {
                self.opt(&n.ins[i])
            } else {
                None
            }
        };
        match n.op.as_str() {
            "Constant" => {
                vec![self.const_tensor(n)]
            }
            "Identity" => vec![g(0).clone()],
            "Add" => vec![elem::binary(g(0), g(1), 0)],
            "Sub" => vec![elem::binary(g(0), g(1), 1)],
            "Mul" => vec![elem::binary(g(0), g(1), 2)],
            "Div" => vec![elem::binary(g(0), g(1), 3)],
            "Max" => vec![elem::binary(g(0), g(1), 4)],
            "Min" => vec![elem::binary(g(0), g(1), 5)],
            "Pow" => vec![elem::binary(g(0), g(1), 6)],
            "Equal" => vec![elem::binary(g(0), g(1), 7)],
            "Greater" => vec![elem::binary(g(0), g(1), 8)],
            "Less" => vec![elem::binary(g(0), g(1), 9)],
            "GreaterOrEqual" => vec![elem::binary(g(0), g(1), 10)],
            "LessOrEqual" => vec![elem::binary(g(0), g(1), 11)],
            "Neg" => vec![elem::unary(g(0), 0)],
            "Exp" => vec![elem::unary(g(0), 1)],
            "Log" => vec![elem::unary(g(0), 2)],
            "Sqrt" => vec![elem::unary(g(0), 3)],
            "Tanh" => vec![elem::unary(g(0), 4)],
            "Sigmoid" => vec![elem::unary(g(0), 5)],
            "Sin" => vec![elem::unary(g(0), 6)],
            "Cos" => vec![elem::unary(g(0), 7)],
            "Abs" => vec![elem::unary(g(0), 8)],
            "Floor" => vec![elem::unary(g(0), 9)],
            "Ceil" => vec![elem::unary(g(0), 10)],
            "Round" => vec![elem::unary(g(0), 11)],
            "Erf" => vec![elem::unary(g(0), 12)],
            "Reciprocal" => vec![elem::unary(g(0), 13)],
            "Cast" => {
                let to = Dt::from_onnx(n.attr_i("to").expect("cast to") as i32);
                vec![elem::cast(g(0), to)]
            }
            "Where" => vec![elem::where_(g(0), g(1), g(2))],
            "Clip" => {
                let lo = o(1).map_or(f32::NEG_INFINITY, |t| t.scalar_f());
                let hi = o(2).map_or(f32::INFINITY, |t| t.scalar_f());
                vec![elem::clip(g(0), lo, hi)]
            }
            "LeakyRelu" => vec![elem::leaky(g(0), n.attr_f("alpha").unwrap_or(0.01))],
            "IsNaN" => vec![elem::isnan(g(0))],
            "And" => vec![elem::logical(g(0), g(1), 0)],
            "Or" => vec![elem::logical(g(0), g(1), 1)],
            "Mod" => vec![elem::mod_(g(0), g(1), n.attr_i("fmod").unwrap_or(0) != 0)],
            "MatMul" => vec![matmul::matmul(g(0), g(1))],
            "Gemm" => {
                let a = g(0);
                let b = g(1);
                let alpha = n.attr_f("alpha").unwrap_or(1.0);
                let beta = n.attr_f("beta").unwrap_or(1.0);
                let ta = n.attr_i("transA").unwrap_or(0) != 0;
                let tb = n.attr_i("transB").unwrap_or(0) != 0;
                let av = if ta {
                    shape::transpose(a, &[1, 0])
                } else {
                    a.clone()
                };
                let bv = if tb {
                    shape::transpose(b, &[1, 0])
                } else {
                    b.clone()
                };
                let (m, k) = (av.dims[0], av.dims[1]);
                let nn_ = bv.dims[1];
                let mut out = T::new(Dt::F32, vec![m, nn_]);
                let bias = o(2).map(|t| t.f().to_vec());
                let mut c = vec![0f32; m * nn_];
                matmul::gemm(m, nn_, k, av.f(), bv.f(), &mut c, bias.as_deref(), None);
                if alpha != 1.0 {
                    for v in c.iter_mut() {
                        *v *= alpha;
                    }
                }
                if beta != 1.0 {
                    for v in c.iter_mut() {
                        *v *= beta;
                    }
                }
                out.fm().copy_from_slice(&c);
                vec![out]
            }
            "Conv" => {
                let st = n
                    .attr_is("strides")
                    .map(|v| v.to_vec())
                    .unwrap_or_else(|| vec![1, 1]);
                let pd = n
                    .attr_is("pads")
                    .map(|v| v.to_vec())
                    .unwrap_or_else(|| vec![0, 0, 0, 0]);
                let dl = n
                    .attr_is("dilations")
                    .map(|v| v.to_vec())
                    .unwrap_or_else(|| vec![1, 1]);
                let gr = n.attr_i("group").unwrap_or(1);
                vec![conv::conv(g(0), g(1), o(2), &st, &pd, &dl, gr)]
            }
            "ConvTranspose" => {
                let st = n
                    .attr_is("strides")
                    .map(|v| v.to_vec())
                    .unwrap_or_else(|| vec![1, 1]);
                let pd = n
                    .attr_is("pads")
                    .map(|v| v.to_vec())
                    .unwrap_or_else(|| vec![0, 0, 0, 0]);
                let op = n
                    .attr_is("output_padding")
                    .map(|v| v.to_vec())
                    .unwrap_or_else(|| vec![0, 0]);
                let gr = n.attr_i("group").unwrap_or(1);
                vec![conv::conv_transpose(g(0), g(1), o(2), &st, &pd, &op, gr)]
            }
            "LayerNormalization" => {
                let ax = n.attr_i("axis").unwrap_or(-1);
                let eps = n.attr_f("epsilon").unwrap_or(1e-5);
                vec![nn::layer_norm(g(0), g(1), g(2), ax, eps)]
            }
            "InstanceNormalization" => {
                let eps = n.attr_f("epsilon").unwrap_or(1e-5);
                vec![nn::instance_norm(g(0), g(1), g(2), eps)]
            }
            "Softmax" => {
                let ax = n.attr_i("axis").unwrap_or(-1);
                vec![nn::softmax(g(0), ax)]
            }
            "Resize" => {
                let scales = o(2).map(|t| t.f().to_vec());
                let sizes = o(3).map(|t| t.ints());
                let mode = n
                    .attr_s("mode")
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| b"nearest".to_vec());
                let sc = scales.unwrap_or_else(|| vec![1.0; g(0).dims.len()]);
                if mode == b"linear" {
                    vec![nn::resize_linear(g(0), &sc, sizes.as_deref())]
                } else {
                    vec![nn::resize_nearest(g(0), &sc, sizes.as_deref())]
                }
            }
            "Reshape" => {
                let s = g(1).ints();
                let x = g(0);
                let mut s2 = s.clone();
                for (i, v) in s2.iter_mut().enumerate() {
                    if *v == 0 {
                        *v = *x.dims.get(i).unwrap_or(&1) as i64;
                    }
                }
                if let Some(i) = s2.iter().position(|&v| v < 0) {
                    let known: i64 = s2.iter().filter(|&&v| v > 0).product();
                    s2[i] = x.len() as i64 / known;
                }
                vec![shape::reshape(x, &s2)]
            }
            "Transpose" => {
                let p = n.attr_is("perm").map(|v| v.to_vec()).unwrap_or_else(|| {
                    let r = g(0).dims.len();
                    (0..r).rev().map(|i| i as i64).collect()
                });
                vec![shape::transpose(g(0), &p)]
            }
            "Slice" => {
                let st = g(1).ints();
                let en = g(2).ints();
                let ax = o(3)
                    .map(|t| t.ints())
                    .unwrap_or_else(|| (0..st.len() as i64).collect());
                let sp = o(4).map(|t| t.ints()).unwrap_or_else(|| vec![1; st.len()]);
                vec![shape::slice(g(0), &st, &en, &ax, &sp)]
            }
            "Gather" => {
                let a = g(0);
                let b = g(1);
                vec![shape::gather(a, b, n.attr_i("axis").unwrap_or(0))]
            }
            "GatherElements" => vec![shape::gather_elements(
                g(0),
                g(1),
                n.attr_i("axis").unwrap_or(0),
            )],
            "Concat" => {
                let ts: Vec<&T> = n.ins.iter().map(|s| self.get(s)).collect();
                vec![shape::concat(&ts, n.attr_i("axis").unwrap_or(0))]
            }
            "Unsqueeze" => {
                let ax = o(1)
                    .map(|t| t.ints())
                    .unwrap_or_else(|| n.attr_is("axes").map(|v| v.to_vec()).unwrap_or_default());
                vec![shape::unsqueeze(g(0), &ax)]
            }
            "Squeeze" => {
                let ax = o(1)
                    .map(|t| t.ints())
                    .unwrap_or_else(|| n.attr_is("axes").map(|v| v.to_vec()).unwrap_or_default());
                vec![shape::squeeze(g(0), &ax)]
            }
            "Expand" => {
                let s = g(1).ints();
                vec![shape::expand(g(0), &s)]
            }
            "Pad" => {
                let pd = g(1).ints();
                let val = o(2).map_or(0.0, |t| t.scalar_f());
                let mode = match n.attr_s("mode") {
                    None => 0u8,
                    Some(b"constant") => 0,
                    Some(b"reflect") => 1,
                    Some(b"edge") => 2,
                    Some(other) => panic!(
                        "Pad {}: unsupported mode {:?}",
                        n.name,
                        String::from_utf8_lossy(other)
                    ),
                };
                vec![shape::pad(g(0), &pd, val, mode)]
            }
            "Range" => vec![shape::range(g(0), g(1), g(2))],
            "CumSum" => vec![shape::cumsum(g(0), g(1).scalar_i())],
            "ReduceSum" => {
                let ax = o(1).map(|t| t.ints()).unwrap_or_default();
                let keep = n.attr_i("keepdims").unwrap_or(1) != 0;
                vec![shape::reduce_sum(g(0), &ax, keep)]
            }
            "Shape" => {
                let d: Vec<i64> = g(0).dims.iter().map(|&v| v as i64).collect();
                let st = n.attr_i("start").unwrap_or(0);
                let en = n.attr_i("end").unwrap_or(d.len() as i64);
                let r = d.len() as i64;
                let s = norm_axis(st, r as usize) as i64;
                let e = norm_axis(en, r as usize) as i64;
                vec![T::from_i64(
                    vec![(e - s) as usize],
                    d[s as usize..e as usize].to_vec(),
                )]
            }
            "ConstantOfShape" => {
                let d = g(0).ints();
                let dims: Vec<usize> = d.iter().map(|&v| v as usize).collect();
                let (dt, fill) = match n.attrs.iter().find(|(k, _)| k == "value") {
                    Some((_, Attr::T(b) | Attr::S(b))) => {
                        let v = crate::onnx::parse_tensor(b);
                        let esz = v.dt.size();
                        (v.dt, v.data[..esz].to_vec())
                    }
                    _ => (Dt::F32, 0f32.to_le_bytes().to_vec()),
                };
                let n_el: usize = dims.iter().product();
                let esz = dt.size();
                let mut data = Vec::with_capacity(n_el * esz);
                for _ in 0..n_el {
                    data.extend_from_slice(&fill);
                }
                vec![T {
                    dt,
                    dims,
                    d: Arc::new(crate::tensor::align_bytes(&data)),
                }]
            }
            "StridedSlice" => {
                let idx = n.attr_i("index").unwrap_or(0);
                vec![crate::fuse::strided_slice(g(0), idx)]
            }
            "SineExcitation" => {
                let thr = n.attr_f("thr").unwrap_or(11000.0);
                let gthr = n.attr_f("gthr").unwrap_or(10.0);
                vec![crate::fuse::sine_excitation(g(0), g(1), g(2), thr, gthr)]
            }
            "RandomNormalLike" => {
                let src = g(0);
                let mut t = T::new(Dt::F32, src.dims.clone());
                let mean = n.attr_f("mean").unwrap_or(0.0);
                let scale = n.attr_f("scale").unwrap_or(1.0);
                let o = t.fm();
                for v in o.iter_mut() {
                    *v = mean + scale * self.rng.normal();
                }
                vec![t]
            }
            "Nop" => Vec::new(),
            "ElemChain" => {
                let op1 = n.attr_i("op1").unwrap_or(0) as u8;
                let op2 = n.attr_i("op2").unwrap_or(0) as u8;
                vec![crate::fuse::elem_chain(g(0), g(1), g(2), op1, op2)]
            }
            "DequantAlias" => {
                // An `Identity` that only aliased a `Dequant` output. Register the
                // alias name against the same int8 payload so the expansion stays
                // lazy for this consumer too, instead of pinning a second f32 copy.
                let src = g(0);
                let s = g(1);
                let z = o(2);
                let per_row = if s.len() == 1 {
                    src.len()
                } else {
                    src.len() / s.len()
                };
                let sf: Vec<f32> = match s.dt {
                    Dt::F16 => s.h().iter().map(|&v| f16_to_f32(v)).collect(),
                    Dt::F32 => s.f().to_vec(),
                    Dt::F64 => {
                        s.d.chunks_exact(8)
                            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                            .collect()
                    }
                    _ => s.ints().iter().map(|&v| v as f32).collect(),
                };
                let zp: Vec<i8> = match z {
                    Some(t) => t.i8s().to_vec(),
                    None => vec![0; s.len()],
                };
                let q = T {
                    dt: src.dt,
                    dims: src.dims.clone(),
                    d: src.d.clone(),
                };
                if let Some(name) = n.outs.first().filter(|s| !s.is_empty()) {
                    self.dequant_src.insert(
                        name.clone(),
                        DequantSrc {
                            q,
                            scale: sf,
                            zp,
                            per_row,
                        },
                    );
                }
                Vec::new()
            }
            "DequantizeLinear" => {
                // Unfused path (FUSE=0): expand eagerly, as before.
                let x = g(0);
                let s = g(1);
                let z = o(2);
                let mut out = T::new(Dt::F32, x.dims.clone());
                let per_row = if s.len() == 1 {
                    x.len()
                } else {
                    x.len() / s.len()
                };
                let sf: Vec<f32> = match s.dt {
                    Dt::F16 => s.h().iter().map(|&v| f16_to_f32(v)).collect(),
                    Dt::F32 => s.f().to_vec(),
                    Dt::F64 => {
                        s.d.chunks_exact(8)
                            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                            .collect()
                    }
                    _ => s.ints().iter().map(|&v| v as f32).collect(),
                };
                match x.dt {
                    Dt::I8 => {
                        let zp: Vec<i8> = match z {
                            Some(t) => t.i8s().to_vec(),
                            None => vec![0; s.len()],
                        };
                        dequant_i8_f32(x.i8s(), &sf, &zp, out.fm(), per_row);
                    }
                    Dt::F16 => dequant_f16(x.h(), out.fm()),
                    Dt::F32 => return vec![x.clone()],
                    _ => panic!("dequant {:?}", x.dt),
                }
                vec![out]
            }
            "Dequant" => {
                // Register the int8 payload; the f32 expansion happens lazily in
                // `eval` when a consumer asks for this output. Nothing is stored in
                // `vals`, so the 31 MB of dequantized weights is never all live.
                let x = g(0);
                let s = g(1);
                let z = o(2);
                let per_row = if s.len() == 1 {
                    x.len()
                } else {
                    x.len() / s.len()
                };
                let sf: Vec<f32> = match s.dt {
                    Dt::F16 => s.h().iter().map(|&v| f16_to_f32(v)).collect(),
                    Dt::F32 => s.f().to_vec(),
                    Dt::F64 => {
                        s.d.chunks_exact(8)
                            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                            .collect()
                    }
                    _ => s.ints().iter().map(|&v| v as f32).collect(),
                };
                let zp: Vec<i8> = match z {
                    Some(t) => t.i8s().to_vec(),
                    None => vec![0; s.len()],
                };
                let q = T {
                    dt: x.dt,
                    dims: x.dims.clone(),
                    d: x.d.clone(),
                };
                if let Some(name) = n.outs.first().filter(|s| !s.is_empty()) {
                    self.dequant_src.insert(
                        name.clone(),
                        DequantSrc {
                            q,
                            scale: sf,
                            zp,
                            per_row,
                        },
                    );
                }
                Vec::new()
            }
            "QuantizeLinear" => {
                let x = g(0);
                let s = g(1);
                let z = o(2);
                let mut out = T::new(Dt::I8, x.dims.clone());
                let sc = s.f();
                let zp = z.map_or(0i8, |t| t.i8s()[0]);
                let xv = x.f();
                let ov = unsafe {
                    std::slice::from_raw_parts_mut(out.dm().as_mut_ptr() as *mut i8, xv.len())
                };
                for i in 0..xv.len() {
                    let s = sc[if sc.len() == 1 {
                        0
                    } else {
                        i / (xv.len() / sc.len())
                    }];
                    ov[i] = ((xv[i] / s).round() as i32 + zp as i32).clamp(-128, 127) as i8;
                }
                vec![out]
            }
            "LSTM" => {
                let key = n.name.clone();
                if !self.lstm_cache.contains_key(&key) {
                    let layout = n.attr_i("layout").unwrap_or(0);
                    let w = nn::dequant_lstm(g(1), g(2), g(3), layout);
                    self.lstm_cache.insert(key.clone(), w);
                }
                let x = g(0).clone();
                let lw = &self.lstm_cache[&key];
                let (seq, batch) = if lw.layout == 0 {
                    (x.dims[0], x.dims[1])
                } else {
                    (x.dims[1], x.dims[0])
                };
                let xin = if lw.layout == 0 {
                    x
                } else {
                    shape::transpose(&x, &[1, 0, 2])
                };
                let (y, yh, yc) = nn::lstm(&xin, lw, seq, batch);
                vec![y, yh, yc]
            }
            "Flatten" => {
                let ax = n.attr_i("axis").unwrap_or(1);
                let x = g(0);
                let r = x.dims.len();
                let a = norm_axis(ax, r);
                let outer: usize = x.dims[..a].iter().product();
                let inner: usize = x.dims[a..].iter().product();
                vec![shape::reshape(x, &[outer as i64, inner as i64])]
            }
            other => panic!("unsupported op {other:?} node={:?} ins={:?}", n.name, n.ins),
        }
    }

    fn const_tensor(&self, n: &Node) -> T {
        for (k, a) in &n.attrs {
            if k == "value" {
                match a {
                    Attr::T(b) | Attr::S(b) => {
                        let t = crate::onnx::parse_tensor(b);
                        return T {
                            dt: t.dt,
                            dims: t.dims,
                            d: t.data,
                        };
                    }
                    Attr::F(v) => return T::from_f32(vec![], vec![*v]),
                    Attr::I(v) => return T::from_i64(vec![], vec![*v]),
                    Attr::Is(v) => return T::from_i64(vec![v.len()], v.clone()),
                    Attr::Fs(v) => return T::from_f32(vec![v.len()], v.clone()),
                }
            }
        }
        panic!("constant without value: {}", n.name);
    }
}

pub fn f16_slice(a: &[u16], o: &mut [f32]) {
    for (x, &y) in o.iter_mut().zip(a.iter()) {
        *x = f16_to_f32(y);
    }
}

/// Expand a `Dequant` node's int8 payload into f32.
fn expand_dequant(src: &DequantSrc) -> T {
    let mut out = T::new(Dt::F32, src.q.dims.clone());
    match src.q.dt {
        Dt::I8 => dequant_i8_f32(src.q.i8s(), &src.scale, &src.zp, out.fm(), src.per_row),
        Dt::F16 => dequant_f16(src.q.h(), out.fm()),
        Dt::F32 => out.dm().copy_from_slice(&src.q.d),
        other => panic!("dequant {other:?}"),
    }
    out
}
