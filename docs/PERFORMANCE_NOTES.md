# Performance notes

What was tried, what helped, what did not, and the numbers behind each verdict.
Organized by topic rather than chronologically.

Machine: AMD Ryzen 7 7735HS (Zen 3+, 8C/16T), 16 GB DDR5-5600, Windows 11,
power scheme "Best performance", background applications closed. Single thread
unless stated. 52-token input, 3.425 s of audio, unless stated.

Absolute milliseconds on this laptop drift by 20%+ between sessions, so every
speed verdict below rests on an interleaved A/B measurement (both binaries
alternating in one session, ratio only) or on a microbenchmark, never on
cross-session milliseconds.

---

## 1. The `/head` vocoder is an STFT, not a convolution

The single largest win. The six `/head/Conv` nodes have a **generated** weight,
not a learned one: a `Cos`/`Sin`/`Mul`/`Concat` subgraph builds it at load time,
and it is a windowed DFT basis.

Four bases, all `win[n] * {cos,sin}(2*pi*k*n/N)`:

| tensor | shape | N | bins | row scale | used by |
|---|---|---|---|---|---|
| `head.st.fwd` | `[602,600]` | 600 | 0..300 | 1 | 6x `Conv` (stride 150) |
| `head.sb.fwd` | `[152,150]` | 150 | 0..75 | 1 | `Conv_6` (stride 75) |
| `head.st.inv` | `[602,600]` | 600 | 0..300 | `[1/N, 2/N, ...]` | `ConvTranspose` |
| `head.sb.inv` | `[152,150]` | 150 | 0..75 | `[1/N, 2/N, ...]` | `ConvTranspose_2` |

Construction, verified against golden to ~1e-9:

```
win[n] = sin^2(pi * n / N)                       n = 0..N-1
rows 0..N/2      = win[n] * cos(2*pi*k*n/N)      k = 0..N/2
rows N/2+1..N    = -win[n] * sin(2*pi*k*n/N)     k = 0..N/2
```

The layout is **blocked, not interleaved**: all cosines first, then all negated
sines. The window is `sin^2(pi*n/N)` with period **N**, not `N-1`: `hann600`
matches to 1.4e-7, `hann599` is off by 4.0e-3.

So the six convs are a **windowed DFT of length 600, hop 150, 301 bins** over an
82800-sample signal, 549 output positions each. `rfft(seg * win)` matches the
golden `/head/Conv` output to **5.3e-7**.

| method | time | vs naive |
|---|---:|---:|
| naive conv | 490 ms each | 1x |
| numpy `W @ seg` per position | 11.5 ms | 43x |
| 549 separate `rfft(600)` | 5.68 ms | 86x |
| **batched `rfft(600)` over strided frames** | **3.62 ms** | **135x** |

6 convs: **2940 ms -> 21.7 ms**.

`N = 600 = 2^3 * 3 * 5^2`, so a mixed-radix FFT is required. Zero-padding to
1024 was measured at 4.48 ms — slower than the exact 600-point transform, and
numerically wrong unless the window is padded too. Do not do it.

The six convs share the same weight but have **different inputs** (`Pad_0`..
`Pad_5`), so no FFT can be shared between them.

**A false lead.** An early analysis claimed the weight was a plain unwindowed DFT
basis (residual 3.9e-15). That was a square 602x602 exact solve, not a
regression, and an earlier attempt to recover `W` by least squares from only 20
output positions was underdetermined (600 unknowns), giving a meaningless
`||W - win*[C;S]||_max = 1.64`. A direct test settled it: `rfft(seg*win)`
matches golden to 5.3e-7.

### ConvTranspose

`/head/ConvTranspose` is a GEMM followed by a strided overlap-add:

```
G[i, k] = sum_ic X[ic, i] * W[ic, k]        # [549,602] @ [602,600]
out[i*150 .. i*150+600] += G[i]
```

Every output position is written by 1-4 `(i,k)` pairs (mean 3.98), so
accumulation is mandatory and the output must be zero-initialised. Verified
reconstruction error 1.55e-7. **907 ms -> 2.38 ms** (381x).

`/head/ConvTranspose_1` and `_3` take an all-ones input (`ConstantOfShape`), so
their output is just the window `head.st.w2` tiled with stride 150 and
overlap-added. Verified error 0.0. No GEMM at all.

### Folding the basis generation

The four basis matrices are built at run time by ~80 `Cos`/`Sin`/`Mul`/`Mod`/
`Concat` nodes, costing ~19 ms per run. They are a pure function of the kernel
geometry, so they are bit-identical for every input length (verified against all
three golden dirs).

Folding them into initializers is a **load/synth trade, not a speedup**:

| | off (default) | on | delta |
|---|---:|---:|---:|
| loaded RSS | 15.6 MB | 18.8 MB | +3.2 MB |
| peak RSS | 43.1 MB | 44.9 MB | +1.8 MB |
| synth | 234.3 ms | 219.5 ms | -14.8 ms |

For a one-shot process the extra load time cancels the saved synth time. It is
therefore **off by default** and opt-in via `FOLD_BASIS=1` for a long-lived
process that loads once and synthesizes many times.

---

## 2. SGEMM

The core of Conv, ConvTranspose and MatMul.

### Microkernel choice

AVX2 gives 16 YMM registers. Three candidate schemes, measured with one tile,
K=4096, panels L1-resident, 3000 reps:

| scheme | accumulators | B vectors | broadcast | total | GFLOP/s |
|---|---:|---:|---:|---:|---:|
| 8x8 | 8 | 1 | 1 | 10 | 102.4 |
| **6x16** | **12** | **2** | **1** | **15** | **146.4** |
| 4x24 | 12 | 3 | 1 | 16 | 142.0 |

**6x16 wins.** Not on flop count — all three issue 32 flops/cycle at peak — but
on **scheduling headroom**. With 8 accumulators the compiler has 8 free
registers but needs them for the next iteration's loads and broadcasts, so it
cannot hoist far enough ahead and the FMA ports stall. With 12 accumulators
there are 4 free registers, exactly enough for 2 B loads and 1 broadcast of the
next k-step. The generated assembly confirms it: unrolled by 2, issuing 12 FMAs
+ 2 `vmovups` + 6 `vbroadcastss` per k-step with no spills.

Per k-step: 12 FMA at 2/cycle = 6 cycles. 12 A x 8 lanes x 2 flops = 192 flops /
6 cycles = **32 flops/cycle = the AVX2+FMA peak**. The 8 loads fit in Zen 3+'s 3
load ports (2.67 needed), so the FMA ports are the only bottleneck.

4x24 is within 3% but has a worse M-edge (M=137 -> 34 tiles + 1 scalar row, vs
6x16's 22 tiles + 5 rows) and needs 3 B loads per k-step.

**C must be loaded, not zeroed.** The driver calls the microkernel once per
k-block; a kernel starting from `_mm256_setzero_ps()` keeps only the last
block's contribution. This was a real bug, producing `max_abs` errors of ~40.

### Packing

A panel is `MC x KC` row-major with rows padded to a multiple of 8. B panel is
`KC x NC` stored as 8-wide column strips, so the 16 values for a fixed `p` are
two contiguous 32-byte loads instead of 16 strided 4-byte gathers.

**Padding must be per-block, not global.** `kc_pad` and `nc_pad` are
`(kc+7)&!7` for the block being packed, not the global `KC`/`NC`. Using the
global value makes the microkernel read the next row's data past the end of a
short final k-block. This was the second real bug, with the same ~40 magnitude.

Packing costs 0.0363 ms on M274 N256 K966 — 2.2% of the kernel's 1.615 ms.
Across nine measured shapes, packed beats direct on 6 of 9:

| shape | packed 6x16 | packed 8x8 | direct 8x8 | winner |
|---|---:|---:|---:|---|
| M274 N256 K966 | **83.91** | 75.85 | 76.41 | packed 6x16 |
| M137 N322 K966 | 53.60 | 61.93 | **67.27** | direct |
| M274 N96 K288 | 64.42 | 55.76 | **81.02** | direct |
| M137 N256 K322 | 52.03 | 72.06 | **74.27** | direct |
| M274 N192 K576 | 80.26 | 77.52 | **88.24** | direct |
| M137 N192 K576 | 58.46 | 80.13 | **82.24** | direct |
| M274 N160 K1120 | 84.44 | 84.55 | **85.77** | direct |
| M512 N512 K512 | **95.29** | 89.47 | 81.73 | packed 6x16 |
| M1024 N1024 K1024 | **93.65** | 87.49 | 52.93 | packed 6x16 |

The pattern: **packing pays when K is large, loses when K is small.** At K=288
the whole B matrix is 110 KB, fits in L2, and the strided access is cheap, so
packing is pure overhead. At K=1024 B is 4 MB and the strided access thrashes;
packing wins by 1.77x.

**Decision:** pack when `K >= 768`, else use the direct kernel. A later
measurement lowered the threshold to `K >= 256` and gained 6.5% end to end.

### Blocking

For a `(MC, NC, KC)` block the inner loop touches `6*KC*4` bytes of A and
`KC*16*4` of B. At KC=128 that is 3 KB + 8 KB = 11 KB, comfortably inside the
64 KB L1d. KC=256 costs 45%, KC=64 costs 55%.

Final constants: 6x16 with MC=96 NC=192 KC=128; 8x8 with MC=128 NC=256 KC=128.

### Achieved

52-95 GFLOP/s on the model's shapes, 1.5x-2.4x over the naive kernel. The
141-146 GFLOP/s figures above are a single L1-resident tile that never touches
memory; the full kernel is well below that.

---

## 3. Memory

Peak RSS went from 417 MB to 46 MB. Four changes, in order of size.

### Liveness-based freeing

One pass before the run loop computes the last consumer index of every tensor.
After each node runs, any input whose last use was this step is removed from
`vals`. **Peak RSS 417 -> 126 MB.**

### Lazy dequantization

The 105 `DequantizeLinear` nodes produce 7.77M f32 elements (31 MB). The
topological order puts every weight-only node first (steps 1034-1106) while
their consumers run at 1663-2417, so 76 of them are live at once: 28.9 MB, 62%
of the live peak.

`Dequant` keeps the int8 payload resident and expands to f32 on first use, then
the run loop drops the f32 right after the consumer has run. **Peak LIVE 48.5 ->
27.5 MB.**

### Refcounted tensor data

`T.d` became `Arc<Vec<u8>>`, so views (`Reshape`, `Squeeze`, `Unsqueeze`,
`Identity`) and the initializer copies in `vals` are a pointer bump rather than a
buffer copy. Mutation goes through `Arc::make_mut`, which clones only when the
buffer is shared. **Loaded RSS 31.8 -> 15.4 MB.**

### Lazy `Dequant` -> `Identity` aliases

A `Dequant` whose only consumer is an `Identity` is folded into that Identity.
The Identity is a pure pass-through, but as a separate node it pinned the
expanded f32 weight in `vals` from the Identity's step until the MatMul's — 30
of them, 0.786 MB each, live for ~600 steps. **Peak LIVE 18.8 -> 13.3 MB.**

### Chunking

Splitting long input at sentence boundaries caps peak RSS at the single-chunk
level. 512-token input: **205.8 MB -> 65.0 MB**, and time to first audio
**2160 ms -> 370 ms**. It costs total synth time (2285 vs 2160 ms) and lengthens
the audio slightly (32.12 s vs 30.30 s), because each chunk is a separate
utterance and the duration predictor gives it its own rhythm. Opt-in above 100
tokens.

### Dropped: chunking inside `/head`

Implemented, produced the right length, cut the 512-token peak from 208 to
46 MB — but the waveform was wrong (correlation 0.83 / 0.50 / -0.01). The
vocoder has length-dependent `Shape`/`Gather`/`Slice` operations with absolute
indices. Reverted.

### Dropped: size-classed free list

A size-classed allocator pool measured 0.9919 on the A/B ratio (a 0.8% win) but
raised peak RSS from 30.7 to 37.2 MB on the short input and 164.9 to 175.0 MB on
the long one. Not worth it.

### Measurement trap

Peak RSS must be read in a **fresh process**. Measuring it in the same process
as an earlier run reports the earlier run's high-water mark. An early version of
these numbers was wrong by 187 MB for exactly this reason. The real churn at 52
tokens is 17.3 MB, not 187 MB.

---

## 4. Elementwise and broadcast

200 `Mul` and 196 `Add` nodes, most of them broadcasting a small tensor against a
large one.

**Fast paths for the common broadcast shapes** (scalar, per-row, per-column,
same-shape) took the elementwise total from 1092 to 721 ms.

**Fusing `Mul`/`Add` chains** into one `ElemChain` node removes the intermediate
from `vals` and runs the loop once instead of twice. 89 chains in this graph.
**3.9% end to end.**

Two bugs worth recording:

- The scalar fast path checked `len() != 1` instead of `len() == n`, causing an
  index-out-of-bounds at len 192.
- `ElemChain` always emitted F32, which broke the integer `Add` at `/bert/Add`.
  Guarded with a dtype check that falls back to the general path.

**Dropped: in-place elementwise.** Writing the result over the first operand
measured 1.0023 on the A/B ratio — a 0.2% loss, within noise. The extra
aliasing complexity is not worth it.

**Dropped: a transpose fast path.** A specialized path covering 84 of 86
transpose nodes measured **1.0098 (1.0% slower)**. A microbenchmark confirmed
0.85x-1.08x — a wash. Root cause: these transposes are **memory-bound, not
index-bound** (43,840 elements = 175 KB, L2-resident), and the compiler had
already strength-reduced the div/mod. Reverted; `runtime/examples/trbench.rs`
is kept as the evidence.

---

## 5. LSTM

Six quantized LSTMs, 49.4 ms originally.

**Transposing the weights at load** to `[dirs, input, 4h]` and `[dirs, h, 4h]`
makes the gate loop read contiguously. ONNX stores `W` as `[dirs, 4h, input]`.
**49.4 -> 13.7 ms.**

ONNX gate order is `i, o, f, c`; `B` is `Wb || Rb`; a backward direction writes
at `ti = seq-1-t`.

The LSTM is still the worst op relative to ORT (4.8x slower) and sits at 10% of
peak, because the gate loop is scalar. Batching the input projection into one
GEMM and SIMD-ing the gates is the obvious next step.

---

## 6. FFT

A mixed-radix (2/3/5) Stockham autosort with a reusable `Plan` that caches
twiddles.

- A recursive DIT with an O(n) copy per level measured 28.9 ms. Stockham
  autosort replaced it.
- Radix-5 initially failed at n=10: missing imaginary cross terms.
- The real-input trick turns 600 real samples into 300 complex, halving the
  work. The unpack step `Z[m-k]` is wrong at `k=m` and must use `Z[0]`; caught
  by `tests/fft.rs` against a naive DFT.

---

## 7. Load time

Load is 12.1 ms against ORT's 610.5 ms.

**`fuse_elem_chain` was O(n^2).** It rebuilt the consumer map over all 2529
nodes on every iteration, 89 times: 48.6 ms. Building the map once and
maintaining it incrementally took it to 1.2 ms. **Load 51.3 -> 9.5 ms.**

**Dropped: constant-folding the DFT basis.** Measured 0.9920 on the A/B ratio
(0.8% faster) but added 4 MB to loaded RSS and 8.6 MB to peak. Dropped at the
time; later re-measured as a load/synth trade and made opt-in (section 1).

---

## 8. Roofline

Sustained SGEMM at 1024^3 measures **73-99 GFLOP/s** on this machine. The
observed clock under load spans **4.37-4.77 GHz** across sessions, measured two
ways (a dependent FMA chain and a dependent integer-add chain). Percentages of a
theoretical peak are therefore not quoted: they move with thermal state, and a
single reading is not reproducible.

---

## 9. Where the time goes

Wall-time accounting, 52-token input, single thread, one timer per phase so the
categories sum to the measured wall within 5%:

| category | ms | share |
|---|---:|---:|
| load: read + parse | 6.4 | 2.4% |
| load: topo + fuse | 2.8 | 1.0% |
| exec init | 0.1 | 0.0% |
| **synth** | **259.0** | **96.6%** |
| output | 0.0 | 0.0% |
| **wall** | **268.3** | |

Synth splits into kernel 251.1 ms (96.9%), dispatch 6.6 ms (2.5%), lazy dequant
6.8 ms (2.6%). One run allocates 275 MB across 26,561 allocations and takes
34,158 page faults. Conv alone accounts for 77 MB in 10,044 allocations (~205
per node: im2col panel, B matrix, C buffer, SGEMM packing).

**Measurement trap:** timing every node with `Instant::now()` costs ~7 us per
node, 14-25 ms across 2521 nodes. That overhead is distributed by node count, so
it inflates ops with many tiny nodes (`Tanh` 192x, `Pow` 64x, `Pad` 33x). The
accounting harness times only ops above a cost threshold and reports the untimed
remainder separately.

---

## 10. Against ONNX Runtime

52-token input, single thread. ORT's per-op proportions scaled to its real
96.7 ms wall; ours from the accounting pass.

| op | ORT ms | ours ms | deficit | ratio |
|---|---:|---:|---:|---:|
| Conv | 29.5 | 71.0 | -41.4 | 2.40 |
| Mul | 13.6 | 31.3 | -17.6 | 2.29 |
| Sin | 8.8 | 25.0 | -16.3 | 2.86 |
| Add | 2.2 | 17.1 | -14.9 | 7.86 |
| MatMul | 9.6 | 21.7 | -12.1 | 2.26 |
| Transpose | 1.4 | 12.4 | -11.1 | 9.17 |
| LSTM | 2.4 | 11.5 | -9.1 | 4.81 |
| ConvTranspose | 8.4 | 12.2 | -3.8 | 1.46 |
| Gather | 1.7 | 5.1 | -3.4 | 3.06 |

The top 7 account for 122 of the 158 ms deficit. `Sin` is mostly our
`SineExcitation` (20.2 ms) plus 7 real `Sin` nodes (4.5 ms); ORT does the same
harmonic sum in 8.8 ms.

**ORT's profiler is not usable for absolute numbers.** Enabling it inflates the
wall clock from 96.1 to 147.8 ms (1.54x) and the kernel sum comes to 12x the
wall. Per-op proportions are stable across runs to about 10%, so they are usable
when scaled to the real wall.

---

## 11. Measurement harness

Several early verdicts were wrong because the harness was wrong. The fixes:

- **`ab_bench.py` compared a binary against itself** for every ratio before the
  fix: `--b-commit HEAD` resolved to the last commit rather than the working
  tree, the worktree reused a stale checkout, and the worktree target dir was
  not cleared so cargo skipped the rebuild. Every pre-fix A/B ratio was invalid
  and had to be re-measured. The harness now prints the resolved commit and
  binary size for each side.
- **Noise floor** after the fix: p50 0.9990, p95 1.0026, i.e. +-0.5%.
- **`bench_stats.py` read a stale binary** because it did not honour
  `CARGO_TARGET_DIR`.
- **ORT RSS readout returned 0** because `PROCESS_MEMORY_COUNTERS` must be
  exactly 10 fields (72 bytes).
- **`livetrace` ran FREE=0 then FREE=1 in one process**, producing a false
  "187 MB churn". One configuration per process.

---

## 12. What is left

Ranked by the deficit table in section 10:

1. **Conv** (-41.4 ms). The 43 non-STFT convs run at 36 GFLOP/s against 98
   GFLOP/s sustained SGEMM. Small-M shapes are the problem: the 6x16 kernel
   wants M >= 96 and most convs have M in the 100-300 range with a poor edge.
2. **Elementwise** (-32.5 ms across Mul and Add). Memory-bound; needs fusion
   across more than two nodes, or in-place with a proper liveness proof.
3. **SineExcitation** (-16.3 ms). 20.2 ms for 5.26M sin evaluations. A vectorized
   `sin` approximation would help, but the phase path must stay f32/f64.
4. **MatMul** (-12.1 ms). Same small-M problem as Conv.
5. **Transpose** (-11.1 ms). Measured memory-bound; needs a layout change
   upstream rather than a faster transpose.
6. **LSTM** (-9.1 ms). Batch the input projection into one GEMM, SIMD the gates.
7. **Threading.** Last, and probably necessary: the 5.19 GFLOP of flop-bound ops
   takes 125 ms on one thread. Reaching 90 ms total needs those ops roughly
   twice as fast *and* everything else in ~20 ms.
