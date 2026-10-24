# Architecture

A from-scratch ONNX runtime for one model. No dependencies: the protobuf reader,
graph loader, tensor type and every op kernel are in this repository.

```
runtime/src/
  pb.rs        protobuf wire format
  onnx.rs      ONNX graph parser
  graph.rs     Model: name interning, producer/consumer maps, topological sort
  tensor.rs    T: dtype + dims + refcounted bytes
  quant.rs     fp16<->fp32, int8 dequant, xorshift RNG, sigmoid, erf
  exec.rs      op dispatch, the run loop, liveness freeing, lazy dequant
  fuse.rs      graph-level pattern rewrites, applied once at load
  alloc.rs     counting global allocator (measurement only)
  kernels/
    elem.rs    elementwise and broadcast
    shape.rs   Reshape, Transpose, Slice, Gather, Concat, Pad, ...
    matmul.rs  GEMM driver, bias and activation
    sgemm.rs   AVX2+FMA microkernels, scalar fallback
    conv.rs    Conv, ConvTranspose, im2col, the STFT path
    nn.rs      LSTM, normalization, Softmax, Resize
    fft.rs     mixed-radix FFT with a reusable Plan
```

## Loading

`onnx::parse` walks the protobuf by hand. Two details matter:

- **Packed repeated fields carry a byte count, not an element count.** A
  repeated field may also arrive unpacked, one tag and value at a time, so the
  reader dispatches on `(field_number, wire_type)` rather than assuming either
  form.
- **`raw_data` has no alignment guarantee.** The typed views in `tensor.rs`
  reinterpret the byte buffer as `f32`/`i64`/`f64`, so every buffer is copied
  through `Vec<u64>` to get alignment 8. A `Vec<u8>` only guarantees alignment
  1, and reading a 1-aligned buffer as `i64` is undefined behaviour.

`graph::Model::new` then runs the rewrite passes, and `Model::from_graph` builds
the name interning, producer/consumer maps and topological order. Node names are
interned to indices so the run loop compares `usize`, not strings.

## Graph rewrites

Applied once at load, in `fuse::rewrite`. `FUSE=0` disables all of them, which
the differential test uses to check the unfused graph still matches ORT.

**1. Sine excitation.** The `/head` generator is a 7-node chain over a
`[1, 64, L]` tensor — 5.26M elements at 54 tokens, 21 MB per intermediate. It
becomes one `SineExcitation` node:

```
f0[t]    = Gather_7[t]                    # pitch
phase[t] = Add_3[t]                       # cumulative phase
out[t]   = f32(f0[t] > 10) * sum_{h=0..63} sin(freq[h] * phase[t])
                            * f32((h+1) * f0[t] < 11000)
```

The harmonic cutoff is hoisted out of the inner loop: for a given `f0[t]` the
last contributing harmonic is `ceil(11000 / f0[t] - 1)`, so the sum runs to that
instead of always to 63.

**2. Transpose + Gather.** `Transpose([0,2,3,1])` followed by 8 constant-index
`Gather(axis=1)` nodes materializes `[1,549,8,301] -> [1,8,301,549]` (1.32M
elements) only to slice it apart again. It becomes 8 strided reads of the
original tensor.

**3. DequantizeLinear -> Dequant.** The 105 dequant nodes produce 7.77M f32
elements (31 MB). The topological order puts every weight-only node first
(steps 1034-1106) while their consumers run at 1663-2417, so 76 of them are live
at once: 28.9 MB, 62% of the live peak. `Dequant` keeps the int8 payload
resident and expands to f32 on first use, then the run loop drops the f32 right
after the consumer has run.

**4. Elementwise chains.** A `Mul` feeding an `Add` over the same shape becomes
one `ElemChain` node, so the intermediate never reaches `vals` and the loop runs
once instead of twice. 89 chains in this graph.

**5. DFT-basis folding** (opt-in, `FOLD_BASIS=1`). The four `/head` basis
matrices are built at run time by ~80 `Cos`/`Sin`/`Mul`/`Mod`/`Concat` nodes.
They are a pure function of the kernel geometry, so they are bit-identical for
every input length. Folding evaluates the closure once at load and replaces it
with four initializers. Off by default: it is a load/synth trade, not a
speedup (see `PERFORMANCE_NOTES.md`).

## The run loop

`Exec::run` walks `Model::order` and dispatches on the op name. Three things
make it cheap:

**Liveness freeing.** Before the loop, one pass computes the last consumer index
of every tensor. After each node runs, any input whose last use was this step is
removed from `vals`. This is what keeps peak RSS at 79 MB instead of 417 MB.

**Lazy dequantization.** A `Dequant` node stores its int8 payload plus scale and
zero-point. `Exec::eval` expands it to f32 only when a consumer actually needs
it, and drops the f32 again immediately after. The 31 MB of dequantized weights
is therefore never all live at once.

**Refcounted tensor data.** `T.d` is an `Arc<Vec<u8>>`, so views (`Reshape`,
`Squeeze`, `Unsqueeze`, `Identity`) and the initializer copies in `vals` are a
pointer bump rather than a buffer copy. Mutation goes through `Arc::make_mut`,
which clones only when the buffer is shared.

## Kernels

**SGEMM** (`sgemm.rs`) is the core of Conv, ConvTranspose and MatMul. Two AVX2+FMA
microkernels, chosen by K:

| K | kernel | blocking |
|---|---|---|
| >= 768 | packed 6x16 | MC=96 NC=192 KC=128 |
| < 768 | unpacked 8x8 | MC=128 NC=256 KC=128 |

Feature detection is `is_x86_feature_detected!` at run time, with a scalar
fallback that stays in the tree as the reference. `tests/sgemm.rs` checks the
vector paths against it on randomized shapes.

**Conv** (`conv.rs`) has three paths. Group=1 convolutions go through im2col
into SGEMM. Depthwise convolutions (`group == cin`, `[160,1,7]` weights) use a
naive kernel, which is fine because they are 4 nodes and 0.39 GFLOP total. The
six `/head` convolutions are not convolutions at all — see below.

**The `/head` STFT.** The six `/head/Conv` nodes have a generated weight, not a
learned one: `win[n] = sin^2(pi*n/N)` times a DFT basis, with row scales
`[1/N, 2/N, ...]`. They are stride-150 windowed DFTs. `rfft(seg * win)` matches
the golden `/head/Conv` output to 5.3e-7, so they are computed with a
mixed-radix FFT instead of a 602x600 matrix multiply. The six share the same
weight but have different inputs, so no FFT can be shared between them.

**FFT** (`fft.rs`) is a Stockham autosort with radix 2/3/5 and a reusable
`Plan` that caches twiddles. `rfft` uses the real-input trick: 600 real samples
become 300 complex, halving the work. `tests/fft.rs` checks it against a naive
DFT.

**LSTM** (`nn.rs`) stores `W` transposed to `[dirs, input, 4h]` and `R` to
`[dirs, h, 4h]` at load, so the gate loop reads contiguously. ONNX gate order is
`i, o, f, c`; `B` is `Wb || Rb`; a backward direction writes at `ti = seq-1-t`.

## Chunked synthesis

The model caps at 512 tokens (~30 s). The CLI splits longer text at sentence
boundaries (token 4 is `.`, token 3 is `,`), synthesizes each chunk through the
same `Exec`, and concatenates with a 10 ms crossfade. Each chunk is
`[0] + ids[a..b] + [0]`, so every chunk is a well-formed utterance.

Chunking caps peak RSS at the single-chunk level and gets the first audio out
after one chunk, but it costs total synth time and lengthens the audio slightly,
because the duration predictor gives each chunk its own rhythm. It is therefore
opt-in above 100 tokens.

Chunking *inside* `/head` was implemented and reverted: it produced the right
length and cut the 512-token peak from 208 to 46 MB, but the waveform was wrong
(correlation 0.83 / 0.50 / -0.01) because the vocoder has length-dependent
`Shape`/`Gather`/`Slice` operations with absolute indices.

## Measurement

`alloc.rs` installs a counting global allocator behind atomics (live bytes, peak
live, total allocations, total bytes). `exec.rs` has an accounting mode that
times only ops above a cost threshold, so the `Instant::now()` overhead lands on
ops that can absorb it rather than being spread over 2521 nodes. The harnesses
in `runtime/examples/` use both. See `BENCHMARKS.md`.
