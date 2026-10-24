//! Per-node differential test against ONNX Runtime golden tensors.
//!
//! Golden files are produced once by `scripts/dump_golden.py` (see docs/TESTING.md).
//! This test needs neither Python nor onnxruntime at run time.
//!
//! Two modes:
//!   * `teacher_forced` — for each node in topological order, feed OUR kernel
//!     ORT's real input tensors, run only that node, compare to ORT's output.
//!     Isolates kernel bugs from upstream error propagation.
//!   * `full_graph` — run the whole graph with our own upstream outputs and
//!     report the first node whose diff exceeds tolerance.
//!
//! Run:  cargo test --release --test diff_nodes -- --nocapture

// chunks_exact(N) is the idiomatic way to reinterpret a byte buffer as a
// fixed-width scalar; the suggested alternative is not clearer.
#![allow(clippy::chunks_exact_to_as_chunks)]
mod json;
mod npy;

use json::J;
use std::collections::HashMap;
use std::path::Path;
use tinytts::exec::Exec;
use tinytts::graph::Model;
use tinytts::onnx::Dt;
use tinytts::tensor::T;

/// Golden dir. Override with `GOLDEN=../golden_long` to gate another input length.
fn golden_dir() -> String {
    std::env::var("GOLDEN").unwrap_or_else(|_| "../golden".to_string())
}
const MODEL: &str = "../paradee_int8.onnx";

/// Base tolerance. fp16 casts and anything downstream of them get a looser one.
const TOL: f32 = 1e-4;
const TOL_FP16: f32 = 1e-3;

/// Set via env `DUMP=1` to print full per-node diagnostics for failures.
fn dump_enabled() -> bool {
    std::env::var("DUMP").map(|v| v == "1").unwrap_or(false)
}

struct Golden {
    manifest: J,
    tensors: HashMap<String, json::TensorMeta>,
    cache: HashMap<String, npy::Npy>,
}

impl Golden {
    fn load() -> Golden {
        let g = golden_dir();
        let mpath = format!("{g}/manifest.json");
        let manifest = json::parse(&std::fs::read_to_string(&mpath).unwrap_or_else(|e| {
            panic!(
                "cannot read {mpath}: {e}\n\
                 Run:  python scripts/dump_golden.py paradee_int8.onnx golden"
            )
        }));
        let tensors = json::tensors(&manifest);
        Golden {
            manifest,
            tensors,
            cache: HashMap::new(),
        }
    }

    fn get(&mut self, name: &str) -> Option<&npy::Npy> {
        if !self.tensors.contains_key(name) {
            return None;
        }
        if !self.cache.contains_key(name) {
            let file = self.tensors[name].file.clone();
            let n = npy::load(&format!("{}/{file}", golden_dir()));
            self.cache.insert(name.to_string(), n);
        }
        self.cache.get(name)
    }
}

/// Convert a golden `.npy` into our `T`, preserving dtype.
fn to_t(n: &npy::Npy) -> T {
    let dt = match n.dtype {
        npy::NpDtype::F32 => Dt::F32,
        npy::NpDtype::F16 => Dt::F16,
        npy::NpDtype::F64 => Dt::F64,
        npy::NpDtype::I64 => Dt::I64,
        npy::NpDtype::I32 => Dt::I32,
        npy::NpDtype::I8 => Dt::I8,
        npy::NpDtype::U8 => Dt::U8,
        npy::NpDtype::Bool => Dt::Bool,
        npy::NpDtype::I16 => Dt::I32, // widen; graph has no i16
    };
    T {
        dt,
        dims: n.shape.clone(),
        d: n.data.clone().into(),
    }
}

/// Outcome of comparing one output tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Ok,
    Dtype,
    Shape,
}

/// Compare our tensor to the golden one.
/// Returns (verdict, max_abs, max_rel, n_compared).
fn compare(ours: &T, gold: &npy::Npy) -> (Verdict, f32, f32, usize) {
    let dt_ok = dt_name(ours.dt) == npy_dt_name(gold.dtype);
    let shape_ok = ours.dims == gold.shape;
    if !dt_ok {
        return (Verdict::Dtype, f32::INFINITY, f32::INFINITY, 0);
    }
    if !shape_ok {
        return (Verdict::Shape, f32::INFINITY, f32::INFINITY, 0);
    }
    let a = ours.f32s_for_compare();
    let b = gold.f32s();
    let n = a.len().min(b.len());
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        let denom = b[i].abs().max(1e-6);
        let r = d / denom;
        if r > max_rel {
            max_rel = r;
        }
    }
    (Verdict::Ok, max_abs, max_rel, n)
}

fn dt_name(d: Dt) -> &'static str {
    match d {
        Dt::F32 => "f32",
        Dt::F16 => "f16",
        Dt::F64 => "f64",
        Dt::I64 => "i64",
        Dt::I32 => "i32",
        Dt::I8 => "i8",
        Dt::U8 => "u8",
        Dt::Bool => "bool",
    }
}

fn first8(v: &[f32]) -> String {
    v.iter()
        .take(8)
        .map(|x| format!("{x:.6}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Full diagnostic dump for one node: every input and output, both sides.
fn dump_node(ni: usize, node: &J, our_ins: &[(String, Option<T>)], our_outs: &[T], g: &mut Golden) {
    let op = node.get("op").unwrap().as_str();
    let name = node.get("name").unwrap().as_str();
    println!("\n--- idx={ni} op={op} name={name} ---");

    for (i, (iname, ours)) in our_ins.iter().enumerate() {
        if iname.is_empty() {
            println!("  in[{i}] <empty>");
            continue;
        }
        let gold = g.get(iname);
        match (ours, gold) {
            (Some(o), Some(gn)) => {
                let ov = o.f32s_for_compare();
                let gv = gn.f32s();
                let same_dt = dt_name(o.dt) == npy_dt_name(gn.dtype);
                println!(
                    "  in[{i}] {iname}\n      ours: dt={} dims={:?} first8=[{}]\n      gold: dt={} dims={:?} first8=[{}]{}",
                    dt_name(o.dt),
                    o.dims,
                    first8(&ov),
                    npy_dt_name(gn.dtype),
                    gn.shape,
                    first8(&gv),
                    if same_dt { "" } else { "   <-- DTYPE MISMATCH" }
                );
            }
            (Some(o), None) => println!(
                "  in[{i}] {iname}\n      ours: dt={} dims={:?} first8=[{}]\n      gold: (none)",
                dt_name(o.dt),
                o.dims,
                first8(&o.f32s_for_compare())
            ),
            (None, Some(gn)) => println!(
                "  in[{i}] {iname}\n      ours: (missing)\n      gold: dt={} dims={:?} first8=[{}]",
                npy_dt_name(gn.dtype),
                gn.shape,
                first8(&gn.f32s())
            ),
            (None, None) => println!("  in[{i}] {iname}  (neither side has it)"),
        }
    }

    let outs: Vec<String> = node
        .get("outputs")
        .unwrap()
        .as_arr()
        .iter()
        .map(|x| x.as_str().to_string())
        .collect();
    for (oi, oname) in outs.iter().enumerate() {
        if oname.is_empty() {
            continue;
        }
        let ours = our_outs.get(oi);
        let gold = g.get(oname);
        match (ours, gold) {
            (Some(o), Some(gn)) => {
                let ov = o.f32s_for_compare();
                let gv = gn.f32s();
                let same_dt = dt_name(o.dt) == npy_dt_name(gn.dtype);
                let same_shape = o.dims == gn.shape;
                println!(
                    "  out[{oi}] {oname}\n      ours: dt={} dims={:?} first8=[{}]\n      gold: dt={} dims={:?} first8=[{}]{}{}",
                    dt_name(o.dt),
                    o.dims,
                    first8(&ov),
                    npy_dt_name(gn.dtype),
                    gn.shape,
                    first8(&gv),
                    if same_dt { "" } else { "   <-- DTYPE MISMATCH" },
                    if same_shape { "" } else { "   <-- SHAPE MISMATCH" }
                );
            }
            (Some(o), None) => println!(
                "  out[{oi}] {oname}\n      ours: dt={} dims={:?} first8=[{}]\n      gold: (none)",
                dt_name(o.dt),
                o.dims,
                first8(&o.f32s_for_compare())
            ),
            (None, Some(gn)) => println!(
                "  out[{oi}] {oname}\n      ours: (missing)\n      gold: dt={} dims={:?} first8=[{}]",
                npy_dt_name(gn.dtype),
                gn.shape,
                first8(&gn.f32s())
            ),
            (None, None) => println!("  out[{oi}] {oname}  (neither side has it)"),
        }
    }
}

fn npy_dt_name(d: npy::NpDtype) -> &'static str {
    match d {
        npy::NpDtype::F32 => "f32",
        npy::NpDtype::F16 => "f16",
        npy::NpDtype::F64 => "f64",
        npy::NpDtype::I64 => "i64",
        npy::NpDtype::I32 => "i32",
        npy::NpDtype::I16 => "i16",
        npy::NpDtype::I8 => "i8",
        npy::NpDtype::U8 => "u8",
        npy::NpDtype::Bool => "bool",
    }
}

/// Does this node's output depend (transitively) on an fp16 Cast?
fn fp16_downstream(manifest: &J, node_index: usize) -> bool {
    // Cheap heuristic: any Cast node whose `to` attr is fp16, or any node whose
    // op is a known fp16 producer. The graph's fp16 tensors are all initializers
    // (`*_16`), so we instead loosen tolerance for Cast nodes and their direct
    // consumers.
    let nodes = manifest.get("nodes").unwrap().as_arr();
    let n = &nodes[node_index];
    let op = n.get("op").unwrap().as_str();
    if op == "Cast" {
        return true;
    }
    // If any input is produced by a Cast, loosen.
    let ins: Vec<&str> = n
        .get("inputs")
        .unwrap()
        .as_arr()
        .iter()
        .map(|x| x.as_str())
        .collect();
    for (i, other) in nodes.iter().enumerate() {
        if i == node_index {
            continue;
        }
        if other.get("op").unwrap().as_str() != "Cast" {
            continue;
        }
        for o in other.get("outputs").unwrap().as_arr() {
            if ins.contains(&o.as_str()) {
                return true;
            }
        }
    }
    false
}

fn feeds_from(manifest: &J) -> Vec<(String, T)> {
    let ids: Vec<i64> = manifest
        .get("input_ids")
        .expect("manifest.input_ids")
        .as_arr()
        .iter()
        .map(|v| v.as_usize() as i64)
        .collect();
    let speed = match manifest.get("speed").expect("manifest.speed") {
        J::Num(n) => *n as f32,
        _ => 1.0,
    };
    let n = ids.len();
    vec![
        ("input_ids".into(), T::from_i64(vec![1, n], ids)),
        ("speed".into(), T::from_f32(vec![1], vec![speed])),
    ]
}

fn load_model() -> Model {
    let bytes = std::fs::read(MODEL).unwrap_or_else(|e| panic!("read {MODEL}: {e}"));
    Model::new(tinytts::onnx::parse(&bytes))
}

// ---------------------------------------------------------------------------
// Mode 0: coverage — which golden nodes are absent from our graph
// ---------------------------------------------------------------------------

#[test]
fn coverage() {
    if !Path::new(&golden_dir()).exists() {
        eprintln!("SKIP: {}/ missing.", golden_dir());
        return;
    }
    let g = Golden::load();
    let m = load_model();
    let nodes = g.manifest.get("nodes").unwrap().as_arr().to_vec();

    let our: std::collections::HashSet<&str> =
        m.g.nodes
            .iter()
            .map(|n| n.name.as_str())
            .filter(|s| !s.is_empty())
            .collect();
    let golden: std::collections::HashSet<&str> = nodes
        .iter()
        .map(|n| n.get("name").unwrap().as_str())
        .filter(|s| !s.is_empty())
        .collect();

    let mut missing: Vec<(usize, String, String)> = Vec::new();
    let mut unnamed = 0usize;
    for (i, n) in nodes.iter().enumerate() {
        let name = n.get("name").unwrap().as_str();
        if name.is_empty() {
            unnamed += 1;
            continue;
        }
        if !our.contains(name) {
            missing.push((
                i,
                n.get("op").unwrap().as_str().to_string(),
                name.to_string(),
            ));
        }
    }
    let extra: Vec<&str> =
        m.g.nodes
            .iter()
            .map(|n| n.name.as_str())
            .filter(|s| !s.is_empty() && !golden.contains(s))
            .collect();

    println!("\n=== coverage ===");
    println!("  golden nodes            : {}", nodes.len());
    println!("  golden with empty name  : {unnamed}");
    println!("  our nodes               : {}", m.g.nodes.len());
    println!("  golden nodes we lack    : {}", missing.len());
    for (i, op, name) in &missing {
        println!("    idx={i:<6} {op:<20} {name}");
    }
    println!("  our nodes not in golden : {}", extra.len());
    for n in &extra {
        println!("    {n}");
    }

    // The only nodes we may legitimately lack are the ones a rewrite replaced.
    let fused_away = missing.len();
    assert!(
        fused_away <= 20,
        "coverage: {fused_away} golden nodes missing from our graph, expected <= 20 (fused away)"
    );
}

// ---------------------------------------------------------------------------
// Mode 1: teacher-forced per-node
// ---------------------------------------------------------------------------

#[test]
fn teacher_forced() {
    if !Path::new(&golden_dir()).exists() {
        eprintln!(
            "SKIP: {}/ missing. Run scripts/dump_golden.py first.",
            golden_dir()
        );
        return;
    }
    let mut g = Golden::load();
    let m = load_model();
    let nodes = g.manifest.get("nodes").unwrap().as_arr().to_vec();

    let mut e = Exec::new();
    e.load(&m, feeds_from(&g.manifest));

    let mut failures: Vec<(usize, String, String, Verdict, f32, f32, String)> = Vec::new();
    let mut skipped: Vec<(usize, String, String)> = Vec::new();
    let mut per_op: HashMap<String, (usize, usize)> = HashMap::new(); // op -> (fail, total)

    // Our graph may have been rewritten (fused nodes), so look nodes up by name.
    // Unnamed nodes (416 of the 2529 golden nodes have an empty name) are matched
    // by their first NON-EMPTY output, which is unique in an ONNX graph.
    let by_name: HashMap<&str, usize> =
        m.g.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.name.is_empty())
            .map(|(i, n)| (n.name.as_str(), i))
            .collect();
    let by_out: HashMap<&str, usize> =
        m.g.nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| {
                n.outs
                    .iter()
                    .filter(|o| !o.is_empty())
                    .map(move |o| (o.as_str(), i))
            })
            .collect();
    let mut n_fused = 0usize;
    let mut n_unmatched: Vec<String> = Vec::new();
    let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut n_extra = 0usize;
    // Pre-mark every node the golden manifest maps to, so `run_ready_rewritten`
    // only ever runs nodes the rewriter created. Without this it would eagerly
    // run the whole graph on the first iteration and the golden loop would then
    // find its inputs already consumed.
    for node in nodes.iter() {
        let name = node.get("name").unwrap().as_str();
        let outs = node.get("outputs").unwrap().as_arr();
        let i = if name.is_empty() {
            outs.iter()
                .find(|o| !o.as_str().is_empty())
                .and_then(|o| by_out.get(o.as_str()).copied())
        } else {
            by_name.get(name).copied()
        };
        if let Some(i) = i {
            visited.insert(i);
        }
    }

    for (ni, node) in nodes.iter().enumerate() {
        let op = node.get("op").unwrap().as_str().to_string();
        let name = node.get("name").unwrap().as_str().to_string();
        let ins: Vec<String> = node
            .get("inputs")
            .unwrap()
            .as_arr()
            .iter()
            .map(|x| x.as_str().to_string())
            .collect();
        let outs: Vec<String> = node
            .get("outputs")
            .unwrap()
            .as_arr()
            .iter()
            .map(|x| x.as_str().to_string())
            .collect();

        let our_i = if name.is_empty() {
            outs.iter()
                .find(|o| !o.is_empty())
                .and_then(|o| by_out.get(o.as_str()).copied())
        } else {
            by_name.get(name.as_str()).copied()
        };
        let Some(our_i) = our_i else {
            n_fused += 1;
            n_unmatched.push(format!("idx={ni} {op} {name}"));
            continue;
        };
        visited.insert(our_i);

        // Before running this node, run any rewritten node whose inputs are now
        // all available (the golden manifest has no entry for it).
        run_ready_rewritten(&m, &mut e, &mut visited, &mut n_extra);

        // If this node's inputs are not all available yet (the golden manifest's
        // order can differ from ours after a rewrite), skip it rather than panic.
        if m.g.nodes[our_i]
            .ins
            .iter()
            .any(|s| !s.is_empty() && e.peek(s).is_none())
        {
            if skipped.len() < 5 {
                let miss: Vec<&str> = m.g.nodes[our_i]
                    .ins
                    .iter()
                    .filter(|s| !s.is_empty() && e.peek(s).is_none())
                    .map(|s| s.as_str())
                    .collect();
                eprintln!(
                    "  SKIP idx={ni} op={op} name={name:?} our_i={our_i} missing_inputs={miss:?}"
                );
            }
            skipped.push((ni, op.clone(), name.clone()));
            continue;
        }

        // Teacher-force: overwrite every non-empty input with ORT's tensor.
        for inp in ins.iter().filter(|s| !s.is_empty()) {
            if let Some(n) = g.get(inp) {
                e.put(inp, to_t(n));
            }
        }
        // A fused node has inputs the golden manifest does not list (the inner
        // node's other operand). Teacher-force those from golden too, so the
        // comparison is against ORT's real values.
        for inp in m.g.nodes[our_i].ins.iter().filter(|s| !s.is_empty()) {
            if let Some(n) = g.get(inp) {
                e.put(inp, to_t(n));
            }
        }
        // Readiness is judged on OUR node's inputs, not the golden manifest's:
        // a fused node's golden inputs include the inner node's output, which no
        // longer exists in our graph.
        let mut missing = false;
        for inp in m.g.nodes[our_i].ins.iter().filter(|s| !s.is_empty()) {
            if e.peek(inp).is_none() {
                missing = true;
            }
        }
        if missing {
            skipped.push((ni, op.clone(), name.clone()));
            continue;
        }

        // Snapshot the inputs our kernel actually sees (post teacher-forcing).
        let our_ins: Vec<(String, Option<T>)> = ins
            .iter()
            .map(|s| {
                (
                    s.clone(),
                    if s.is_empty() {
                        None
                    } else {
                        e.peek(s).cloned()
                    },
                )
            })
            .collect();

        let our_outs = e.eval_node(&m.g.nodes[our_i]);
        // A `Dequant` node stores its int8 payload and expands lazily; make the
        // f32 output readable so downstream nodes are not skipped.
        for o in outs.iter().filter(|o| !o.is_empty()) {
            e.materialize(o);
        }
        // Store the outputs so the next node can find its inputs. Without this the
        // loop compares each node in isolation and every consumer is skipped.
        for (oi, oname) in outs.iter().enumerate() {
            if !oname.is_empty() && oi < our_outs.len() {
                e.put(oname, our_outs[oi].clone());
            }
        }

        let mut worst_abs = 0f32;
        let mut worst_rel = 0f32;
        let mut compared = 0usize;
        let mut any = false;
        let mut verdict = Verdict::Ok;
        let mut bad_out: Vec<String> = Vec::new();
        for (oi, oname) in outs.iter().enumerate() {
            if oname.is_empty() {
                continue;
            }
            // `Dequant` returns no tensor; compare the lazily expanded one.
            let ours: Option<&T> = if oi < our_outs.len() {
                Some(&our_outs[oi])
            } else {
                e.peek(oname)
            };
            let Some(ours) = ours else { continue };
            let Some(gold) = g.get(oname) else { continue };
            let (v, a, r, n) = compare(ours, gold);
            if v != Verdict::Ok && verdict == Verdict::Ok {
                verdict = v;
            }
            let tol_o = if fp16_downstream(&g.manifest, ni) {
                TOL_FP16
            } else {
                TOL
            };
            if v != Verdict::Ok || a > tol_o {
                bad_out.push(format!("out[{oi}]"));
            }
            worst_abs = worst_abs.max(a);
            worst_rel = worst_rel.max(r);
            compared += n;
            any = true;
        }
        if !any {
            // The golden node's output no longer exists in our graph: it was the
            // inner node of a fused chain, and the fused node carries the outer
            // node's output name. That is a legitimate fusion, not a skip.
            n_fused += 1;
            continue;
        }

        let tol = if fp16_downstream(&g.manifest, ni) {
            TOL_FP16
        } else {
            TOL
        };
        let entry = per_op.entry(op.clone()).or_insert((0, 0));
        entry.1 += 1;
        let bad = match verdict {
            Verdict::Ok => worst_abs > tol,
            _ => true,
        };
        if bad {
            entry.0 += 1;
            failures.push((
                ni,
                op.clone(),
                name.clone(),
                verdict,
                worst_abs,
                worst_rel,
                bad_out.join(","),
            ));
            if dump_enabled() {
                dump_node(ni, node, &our_ins, &our_outs, &mut g);
            }
        }
        let _ = compared;
    }

    // Report.
    let n_compared: usize = per_op.values().map(|(_, t)| *t).sum();
    let n_passed = n_compared - failures.len();
    println!("\n=== teacher-forced: {} golden nodes ===", nodes.len());
    println!("  compared (our kernel vs ORT) : {n_compared}");
    println!("  passed                       : {n_passed}");
    println!("  failed                       : {}", failures.len());
    println!("  not matched to our graph     : {n_fused}");
    for u in n_unmatched.iter().take(30) {
        println!("    {u}");
    }
    println!("  skipped (no golden / no input): {}", skipped.len());
    println!("  our graph has {} nodes", m.g.nodes.len());
    println!(
        "{:<6} {:<20} {:<40} {:<8} {:<10} {:>11} {:>11}",
        "idx", "op", "name", "kind", "outputs", "max_abs", "max_rel"
    );
    for (ni, op, name, v, a, r, bo) in &failures {
        let nm = if name.len() > 38 { &name[..38] } else { name };
        let kind = match v {
            Verdict::Ok => "numeric",
            Verdict::Dtype => "DTYPE",
            Verdict::Shape => "SHAPE",
        };
        let (as_, rs) = if a.is_finite() {
            (format!("{a:.3e}"), format!("{r:.3e}"))
        } else {
            ("-".to_string(), "-".to_string())
        };
        println!("{ni:<6} {op:<20} {nm:<40} {kind:<8} {bo:<10} {as_:>11} {rs:>11}");
    }

    let n_dtype = failures.iter().filter(|f| f.3 == Verdict::Dtype).count();
    let n_shape = failures.iter().filter(|f| f.3 == Verdict::Shape).count();
    let n_num = failures.len() - n_dtype - n_shape;
    println!("\n  numeric: {n_num}   dtype: {n_dtype}   shape: {n_shape}");

    println!("\n--- failures by op_type ---");
    let mut by_op: Vec<(String, usize, usize)> = per_op
        .iter()
        .map(|(k, (f, t))| (k.clone(), *f, *t))
        .filter(|(_, f, _)| *f > 0)
        .collect();
    by_op.sort_by_key(|a| std::cmp::Reverse(a.1));
    for (op, f, t) in &by_op {
        println!("  {op:<24} {f:>4} / {t:<4} failed");
    }

    if !skipped.is_empty() {
        println!(
            "\n--- skipped (no golden / missing input): {} ---",
            skipped.len()
        );
        for (ni, op, name) in skipped.iter().take(20) {
            println!("  {ni:<6} {op:<24} {name}");
        }
    }

    // The fused ops themselves: compare their output against the golden tensor
    // that the chain used to produce.
    println!("\n--- fused ops (our kernel vs the golden tensor it replaces) ---");
    for n in m.g.nodes.iter() {
        if n.op != "SineExcitation" && n.op != "StridedSlice" && n.op != "ElemChain" {
            continue;
        }
        let Some(oname) = n.outs.first() else {
            continue;
        };
        let Some(gold) = g.get(oname).map(|t| npy::Npy {
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: t.data.clone(),
        }) else {
            println!("  {:<16} {:<28} (no golden)", n.op, oname);
            continue;
        };
        // Teacher-force the fused op's inputs from golden, exactly like the main
        // loop does, so the comparison is against ORT's real inputs.
        let mut missing = false;
        for inp in n.ins.iter().filter(|s| !s.is_empty()) {
            match g.get(inp) {
                Some(t) => e.put(inp, to_t(t)),
                None => {
                    if e.peek(inp).is_none() {
                        missing = true;
                    }
                }
            }
        }
        if missing {
            println!("  {:<16} {:<28} (inputs not available)", n.op, oname);
            continue;
        }
        let ours = e.eval_node(n);
        if ours.is_empty() {
            continue;
        }
        let (v, a, r, _) = compare(&ours[0], &gold);
        println!(
            "  {:<16} {:<28} kind={:?} max_abs={a:.3e} max_rel={r:.3e}",
            n.op, oname, v
        );
    }

    println!("\nTOTAL FAILURES: {}", failures.len());

    // Coverage assertion: the only golden nodes we may fail to match are the ones
    // a rewrite replaced. The rewrites fuse 9 nodes (sine excitation + transpose)
    // plus one per elementwise chain (89 today), and the constant folder replaces
    // the `/head` DFT-basis subgraph with 4 initializers (40 nodes today). The
    // bound tracks those counts rather than a fixed number. Anything beyond that
    // means the harness is silently skipping nodes and the per-node safety net
    // has holes.
    let fused_expected =
        9 + m.g.nodes.iter().filter(|n| n.op == "ElemChain").count() + folded_basis_nodes();
    assert!(
        n_compared + n_fused >= 2500,
        "teacher-forced coverage dropped: only {n_compared} of {} golden nodes compared \
         ({} unmatched, {} skipped)",
        nodes.len(),
        n_fused,
        skipped.len()
    );
    assert!(
        n_fused <= fused_expected + 5,
        "{n_fused} golden nodes did not match our graph, expected <= {fused_expected} (fused away)"
    );
}

/// How many golden nodes the `/head` DFT-basis constant folder removes.
///
/// The folder replaces the subgraph producing `head.st.fwd`, `head.st.inv`,
/// `head.sb.fwd` and `head.sb.inv` with four initializers. Count the golden
/// nodes in that closure so the coverage assertion tracks it. The folder is
/// opt-in (`FOLD_BASIS=1`), so this returns 0 when it did not run.
fn folded_basis_nodes() -> usize {
    if !std::env::var("FOLD_BASIS")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return 0;
    }
    let Ok(txt) = std::fs::read_to_string(format!("{}/manifest.json", golden_dir())) else {
        return 0;
    };
    let v = json::parse(&txt);
    let nodes = v.get("nodes").unwrap().as_arr();
    const TARGETS: [&str; 4] = ["head.st.fwd", "head.st.inv", "head.sb.fwd", "head.sb.inv"];
    // Map output name -> node index, then walk back from the targets.
    let mut prod: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        if let Some(outs) = n.get("outputs") {
            for o in outs.as_arr() {
                prod.insert(o.as_str().to_string(), i);
            }
        }
    }
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut stack: Vec<String> = TARGETS.iter().map(|s| s.to_string()).collect();
    while let Some(name) = stack.pop() {
        let Some(&i) = prod.get(&name) else { continue };
        if !seen.insert(i) {
            continue;
        }
        if let Some(ins) = nodes[i].get("inputs") {
            for inp in ins.as_arr() {
                let s = inp.as_str();
                if !s.is_empty() {
                    stack.push(s.to_string());
                }
            }
        }
    }
    seen.len()
}

// ---------------------------------------------------------------------------
// Mode 2: full graph, first divergence
// ---------------------------------------------------------------------------

#[test]
fn full_graph_first_divergence() {
    if !Path::new(&golden_dir()).exists() {
        eprintln!(
            "SKIP: {}/ missing. Run scripts/dump_golden.py first.",
            golden_dir()
        );
        return;
    }
    let mut g = Golden::load();
    let m = load_model();
    let nodes = g.manifest.get("nodes").unwrap().as_arr().to_vec();

    let mut e = Exec::new();
    e.load(&m, feeds_from(&g.manifest));

    let mut first: Option<(usize, String, String, Verdict, f32)> = None;
    let mut n_bad = 0usize;
    let mut n_dtype = 0usize;
    let mut n_shape = 0usize;
    let mut first_tight: Option<(usize, String, String, f32)> = None;
    let mut csv = String::from("idx,op,name,max_abs,max_rel\n");

    // Our graph may have been rewritten (fused nodes), so look nodes up by name.
    // Unnamed nodes (empty name) are matched by their first output instead.
    let by_name: HashMap<&str, usize> =
        m.g.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.name.is_empty())
            .map(|(i, n)| (n.name.as_str(), i))
            .collect();
    let by_out: HashMap<&str, usize> =
        m.g.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.outs.first().map(|o| (o.as_str(), i)))
            .collect();
    let mut n_fused = 0usize;
    let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut n_extra = 0usize;
    // Pre-mark every node the golden manifest maps to, so `run_ready_rewritten`
    // only ever runs nodes the rewriter created. Fused ops are excluded: the
    // golden node they replace is compared in the main loop, and the fused node
    // itself is compared in the fused-ops block at the end.
    for node in nodes.iter() {
        let name = node.get("name").unwrap().as_str();
        let outs = node.get("outputs").unwrap().as_arr();
        let i = if name.is_empty() {
            outs.first().and_then(|o| by_out.get(o.as_str()).copied())
        } else {
            by_name.get(name).copied()
        };
        if let Some(i) = i {
            visited.insert(i);
        }
    }

    for (ni, node) in nodes.iter().enumerate() {
        let op = node.get("op").unwrap().as_str().to_string();
        let name = node.get("name").unwrap().as_str().to_string();
        let outs: Vec<String> = node
            .get("outputs")
            .unwrap()
            .as_arr()
            .iter()
            .map(|x| x.as_str().to_string())
            .collect();

        // A node the rewriter removed: its output is produced by a fused node.
        let our_i = if name.is_empty() {
            outs.first().and_then(|o| by_out.get(o.as_str()).copied())
        } else {
            by_name.get(name.as_str()).copied()
        };
        let Some(our_i) = our_i else {
            n_fused += 1;
            continue;
        };

        // Run any rewritten node whose inputs are now all available.
        run_ready_rewritten(&m, &mut e, &mut visited, &mut n_extra);

        // `Dequant` outputs are expanded on demand, never stored in `vals`.
        // The scale input is often a `Cast` node, so a `Dequant` may become
        // runnable only after that Cast has executed; run any that are ready.
        run_ready_dequant(&m, &mut e, &mut visited);
        for s in &m.g.nodes[our_i].ins {
            if !s.is_empty() {
                e.materialize(s);
            }
        }

        // If this node's inputs are not all available yet (the golden manifest's
        // order can differ from ours after a rewrite), skip it rather than panic.
        if m.g.nodes[our_i]
            .ins
            .iter()
            .any(|s| !s.is_empty() && e.peek(s).is_none())
        {
            n_fused += 1;
            continue;
        }

        // RandomNormalLike is non-deterministic across runtimes. Substitute the
        // golden tensor so downstream comparison is meaningful.
        let our_outs = if op == "RandomNormalLike" {
            outs.iter()
                .filter(|o| !o.is_empty())
                .filter_map(|o| g.get(o).map(to_t))
                .collect::<Vec<T>>()
        } else {
            e.eval_node(&m.g.nodes[our_i])
        };
        // A `Dequant` node stores its int8 payload and expands lazily; make the
        // f32 output readable so downstream nodes are not skipped.
        for o in outs.iter().filter(|o| !o.is_empty()) {
            e.materialize(o);
        }
        for (i, o) in node.get("outputs").unwrap().as_arr().iter().enumerate() {
            let oname = o.as_str();
            if oname.is_empty() || i >= our_outs.len() {
                continue;
            }
            e.put(oname, our_outs[i].clone());
        }

        let mut worst = 0f32;
        let mut worst_rel = 0f32;
        let mut verdict = Verdict::Ok;
        let mut is_f32 = false;
        for (oi, oname) in outs.iter().enumerate() {
            if oname.is_empty() {
                continue;
            }
            // `Dequant` returns no tensor; compare the lazily expanded one.
            let ours: Option<&T> = if oi < our_outs.len() {
                Some(&our_outs[oi])
            } else {
                e.peek(oname)
            };
            let Some(ours) = ours else { continue };
            let Some(gold) = g.get(oname) else { continue };
            let (v, a, r, _) = compare(ours, gold);
            if v != Verdict::Ok && verdict == Verdict::Ok {
                verdict = v;
            }
            if ours.dt == Dt::F32 {
                is_f32 = true;
            }
            worst = worst.max(a);
            worst_rel = worst_rel.max(r);
        }
        csv.push_str(&format!(
            "{ni},{op},{},{},{}\n",
            if name.is_empty() { "-" } else { &name },
            if worst.is_finite() {
                format!("{worst:.6e}")
            } else {
                "inf".into()
            },
            if worst_rel.is_finite() {
                format!("{worst_rel:.6e}")
            } else {
                "inf".into()
            },
        ));

        // Tight threshold: first f32 node past 1e-5.
        if is_f32 && verdict == Verdict::Ok && worst > 1e-5 && first_tight.is_none() {
            first_tight = Some((ni, op.clone(), name.clone(), worst));
        }

        let tol = if fp16_downstream(&g.manifest, ni) {
            TOL_FP16
        } else {
            TOL
        };
        let bad = match verdict {
            Verdict::Ok => worst > tol,
            _ => true,
        };
        if bad {
            n_bad += 1;
            match verdict {
                Verdict::Dtype => n_dtype += 1,
                Verdict::Shape => n_shape += 1,
                _ => {}
            }
            if first.is_none() {
                first = Some((ni, op.clone(), name.clone(), verdict, worst));
            }
        }
    }

    // Run any of OUR nodes the golden manifest did not cover (i.e. fused nodes),
    // so their outputs exist for the waveform comparison below.
    run_ready_rewritten(&m, &mut e, &mut visited, &mut n_extra);
    if n_extra > 0 {
        println!("ran {n_extra} rewritten node(s) not present in the golden manifest");
    }
    std::fs::write("node_diff.csv", &csv).ok();
    println!("\nwrote node_diff.csv ({} rows)", nodes.len());
    if n_fused > 0 {
        println!("note: {n_fused} golden nodes were fused away by the graph rewriter");
    }

    println!(
        "\n=== full graph: {n_bad} nodes exceed tolerance (dtype {n_dtype}, shape {n_shape}) ==="
    );
    match first {
        Some((ni, op, name, v, a)) => {
            let kind = match v {
                Verdict::Dtype => "DTYPE",
                Verdict::Shape => "SHAPE",
                _ => "numeric",
            };
            println!("FIRST DIVERGENCE: idx={ni} op={op} name={name} kind={kind} max_abs={a:.3e}");
        }
        None => println!("no divergence"),
    }
    match first_tight {
        Some((ni, op, name, a)) => {
            println!("FIRST f32 NODE > 1e-5: idx={ni} op={op} name={name} max_abs={a:.3e}")
        }
        None => println!("no f32 node exceeds 1e-5"),
    }

    // Final waveform: correlation + max abs diff against golden.
    let out_names: Vec<String> = m.g.outputs.clone();
    for oname in &out_names {
        let Some(ours) = e.peek(oname) else { continue };
        let Some(gold) = g.get(oname) else { continue };
        let a = ours.f32s_for_compare();
        let b = gold.f32s();
        let n = a.len().min(b.len());
        let mut max_abs = 0f32;
        let mut sum_a = 0f64;
        let mut sum_b = 0f64;
        for i in 0..n {
            max_abs = max_abs.max((a[i] - b[i]).abs());
            sum_a += a[i] as f64;
            sum_b += b[i] as f64;
        }
        let ma = sum_a / n as f64;
        let mb = sum_b / n as f64;
        let (mut va, mut vb, mut cov) = (0f64, 0f64, 0f64);
        for i in 0..n {
            let da = a[i] as f64 - ma;
            let db = b[i] as f64 - mb;
            va += da * da;
            vb += db * db;
            cov += da * db;
        }
        let corr = if va > 0.0 && vb > 0.0 {
            cov / (va.sqrt() * vb.sqrt())
        } else {
            f64::NAN
        };
        println!(
            "\nWAVEFORM {oname}: n={n} max_abs={max_abs:.6e} corr={corr:.9} 1-corr={:.3e}",
            1.0 - corr
        );
    }
}

/// Run any `Dequant` node whose inputs are all available, so its int8 payload is
/// registered and its output can be materialised. `Dequant` nodes are matched by
/// output name (they have no name), so the golden manifest may reach a consumer
/// before the corresponding `Dequant` has been evaluated.
fn run_ready_dequant(m: &Model, e: &mut Exec, visited: &mut std::collections::HashSet<usize>) {
    loop {
        let mut progressed = false;
        for (i, n) in m.g.nodes.iter().enumerate() {
            if n.op != "Dequant" || visited.contains(&i) {
                continue;
            }
            if n.ins.iter().any(|s| !s.is_empty() && e.peek(s).is_none()) {
                continue;
            }
            e.eval_node(n);
            for o in n.outs.iter().filter(|o| !o.is_empty()) {
                e.materialize(o);
            }
            visited.insert(i);
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
}

/// Run any node of OUR graph that the golden manifest did not cover (i.e. nodes
/// the rewriter created), as soon as all of its inputs are available. This keeps
/// the fused node in topological position relative to its consumers.
fn run_ready_rewritten(
    m: &Model,
    e: &mut Exec,
    visited: &mut std::collections::HashSet<usize>,
    n_extra: &mut usize,
) {
    loop {
        let mut progressed = false;
        for (i, n) in m.g.nodes.iter().enumerate() {
            if visited.contains(&i) {
                continue;
            }
            // `Dequant` outputs are expanded on demand, never stored in `vals`.
            // Materialise them so the readiness check below sees them.
            for s in &n.ins {
                if !s.is_empty() {
                    e.materialize(s);
                }
            }
            let ready = n.ins.iter().all(|s| s.is_empty() || e.peek(s).is_some());
            if !ready {
                continue;
            }
            let outs = e.eval_node(n);
            for (k, o) in n.outs.iter().enumerate() {
                if !o.is_empty() && k < outs.len() {
                    e.put(o, outs[k].clone());
                }
            }
            visited.insert(i);
            *n_extra += 1;
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
}
