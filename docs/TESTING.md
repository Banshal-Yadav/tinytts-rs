# Testing

The correctness net is a per-node differential test against ONNX Runtime. It
catches a wrong kernel immediately and names the node, rather than leaving you to
bisect a wrong waveform.

## Golden tensors

`scripts/verify/dump_golden.py` runs ONNX Runtime once with every node output exposed as
a graph output, and writes:

- `golden/t<NNNN>.npy` — one file per tensor, named by manifest index
- `golden/manifest.json` — node order, op types, names, input/output tensor
  names, per-tensor dtype and shape, and the fixed input

Tensor files are indexed, **not** named after the tensor, so sanitisation can
never collide (`/F0.1/Mul` vs `/F0/1/Mul`). The real name lives in the manifest.

Initializers are not dumped: they are weights the runtime loads straight from the
model.

Golden data is **not committed** — it is ~3.4 GB for three input lengths.
Regenerate it once:

```bash
python scripts/verify/dump_golden.py paradee_int8.onnx golden          # 52-token
IDS="..." python scripts/verify/dump_golden.py paradee_int8.onnx golden_short
IDS="..." python scripts/verify/dump_golden.py paradee_int8.onnx golden_long
```

The three inputs are fixed ids shared by every benchmark script and the test, so
they agree on what "short", "mid" and "long" mean.

After that the test needs neither Python nor onnxruntime.

## Test modes

```bash
cd runtime
cargo test --release
```

`tests/diff_nodes.rs` has three modes:

**`coverage`** — which golden nodes are absent from our graph. The only ones
allowed to be missing are the ones a rewrite replaced.

**`teacher_forced`** — for each node in topological order, feed *our* kernel
ORT's real input tensors, run only that node, and compare to ORT's output. This
isolates kernel bugs from upstream error propagation: a wrong `Conv` shows up as
a wrong `Conv`, not as a wrong waveform 2000 nodes later.

**`full_graph_first_divergence`** — run the whole graph with our own upstream
outputs and report the first node whose difference exceeds tolerance, plus the
final waveform correlation.

`GOLDEN=../golden_long` selects a different golden dir. Nodes are looked up **by
name**, not by manifest index, because the graph rewriter removes nodes; unnamed
nodes (416 of the 2529) are matched by their first non-empty output, which is
unique in an ONNX graph.

## Tolerances

Base tolerance is **1e-4**, loosened to **1e-3** for `Cast` nodes and their
direct consumers, because the graph's fp16 tensors pass through them.

`RandomNormalLike` is expected to differ: the graph draws its own noise, so the
test substitutes the golden tensor before comparing downstream. It is the only
node that fails in every configuration.

## The gate

```bash
python scripts/verify/gate.py          # all three lengths
python scripts/verify/gate.py short    # one
```

Runs `full_graph` against each golden dir and prints PASS/FAIL with the
correlation. Exit code 0 only if all pass. The threshold is correlation
>= 0.99999.

Current values:

| input | tokens | samples | max abs | correlation |
|---|---:|---:|---:|---:|
| short | 13 | 31800 | 6.548e-05 | 0.999999995 |
| mid | 52 | 82200 | 3.534e-04 | 0.999999947 |
| long | 512 | 727200 | 2.944e-03 | 0.999998015 |

**Every optimization must be re-gated.** If the correlation drops, revert.

## Teacher-forced counts

The counts depend on which rewrites are enabled. All three configurations are
expected to pass:

| configuration | compared | passed | unmatched | our nodes |
|---|---:|---:|---:|---:|
| `FUSE=0 FOLD_BASIS=0` | 2529 | 2528 | 0 | 2529 |
| `FUSE=0` | 2485 | 2484 | 44 | 2485 |
| default (fused) | 2431 | 2430 | 98 | 2521 |
| `FOLD_BASIS=1` | 2391 | 2390 | 138 | 2481 |

The single failure in every row is `RandomNormalLike`. "Unmatched" counts golden
nodes a rewrite replaced: 9 for the sine-excitation and transpose fusions, one
per `ElemChain` (89), and 40 for the DFT-basis fold.

The coverage assertion tracks these counts rather than a fixed number, so a
rewrite that silently starts skipping nodes fails the test instead of passing
quietly.

## Unit tests

| file | what it checks |
|---|---|
| `tests/sgemm.rs` | the AVX2 microkernels against the scalar reference, randomized shapes |
| `tests/fft.rs` | the mixed-radix FFT against a naive DFT, including the real-input trick |
| `tests/elem.rs` | elementwise fast paths against the general strided broadcast path |
| `tests/npy.rs` | the `.npy` reader |
| `tests/json.rs` | the minimal JSON parser |

The naive kernels (`conv_naive`, `conv_transpose_naive`, the scalar SGEMM path)
stay in the tree as oracles. New kernels get a randomized differential test
against them in addition to the ORT gate.

## Determinism

The graph draws its own noise, so two runs on the same input differ in the noisy
parts of the waveform. `--rng-seed N` fixes the xorshift seed, which makes the
output byte-identical run to run:

```bash
./target/release/tinytts ../paradee_int8.onnx --no-chunk --rng-seed 12345 <ids...>
```

This is how the reference WAVs for a refactor are recorded and compared.

## Regression discipline

1. Gate before and after every change.
2. One change per commit.
3. Keep the old kernel as an oracle.
4. Measure before claiming. Every "this is as fast as it can go" needs a roofline
   number and a measured fraction of peak.
5. Speed claims use the interleaved A/B harness, never cross-session
   milliseconds.
6. Never bulk-delete or regex-rewrite lines without committing first and reading
   `git diff`. A cleanup script that removed lines containing `eprintln!` also
   removed 11 kernel match arms.
