# Benchmarks

Every number here was measured on the machine below. Absolute milliseconds on a
laptop drift by 20%+ between sessions with thermal state and background load, so
speed comparisons use an interleaved A/B benchmark that runs both binaries
alternately in one session and reports only the ratio.

## Machine

| | |
|---|---|
| CPU | AMD Ryzen 7 7735HS (Zen 3+), 8C/16T |
| Base / boost | 3.2 GHz / 4.75 GHz |
| SIMD | AVX2 + FMA, no AVX-512 |
| RAM | 16 GB DDR5-5600 |
| L1d / L2 / L3 | 512 KB / 4 MB / 16 MB |
| OS | Windows 11 |
| Power scheme | Best performance |
| Background apps | closed (browser, LM Studio) |

## Head to head

52-token input, 3.425 s of audio at 24 kHz, single thread unless stated.
`scripts/verify/benchmark.py`, 5 runs.

| | ONNX Runtime 1T | ONNX Runtime 4T | this runtime |
|---|---:|---:|---:|
| load | 610.5 ms | — | **7.9 ms** |
| synth (median) | 93.0 ms | 54.9 ms | 371.4 ms |
| real-time factor | 36.8x | 62.4x | 9.2x |
| RSS while loaded | 82 MB | 82 MB | **24 MB** |
| peak RSS | 230 MB | 230 MB | **79 MB** |

Load is **77x faster**, peak memory **2.9x smaller**, synthesis **4.0x slower**
than ORT on one thread.

ORT's own profiler is not usable for absolute numbers: enabling it inflates the
wall clock from 96.1 to 147.8 ms (1.54x) and the kernel sum comes to 12x the
wall. Per-op *proportions* are stable across runs to about 10%, so they are
usable when scaled to the real wall.

## Memory

Fresh process per measurement, one load and one run each.
`scripts/bench/rss_fresh.py`, median of 5.

| input | tokens | audio | loaded RSS | peak LIVE | peak RSS | synth p50 |
|---|---:|---:|---:|---:|---:|---:|
| short | 13 | 0.93 s | 15.6 MB | 13.3 MB | 30.5 MB | 112.9 ms |
| mid | 52 | 3.43 s | 15.8 MB | 25.3 MB | 42.7 MB | 249.0 ms |
| long | 512 | 30.30 s | 15.7 MB | 148.5 MB | 166.1 MB | 1920.9 ms |

"peak LIVE" is the counting allocator's high-water mark; "peak RSS" is the
process working set. The gap between them is allocator and stack overhead.

Loaded RSS is flat at ~15.7 MB: the 8.46 MB model file plus the runtime. Peak
scales with input length because the vocoder's intermediates do.

**Measurement trap:** peak RSS must be read in a fresh process. Measuring it in
the same process as an earlier run reports the earlier run's high-water mark.
An early version of these numbers was wrong by 187 MB for exactly this reason.

## Chunked synthesis

512-token input. `scripts/lab/chunkdefaults.py`.

| mode | chunks | peak RSS | synth | time to first audio | audio |
|---|---:|---:|---:|---:|---:|
| `--no-chunk` | 1 | 205.8 MB | 2160.4 ms | 2160.4 ms | 30.30 s |
| default (auto) | 7 | 65.0 MB | 2285.5 ms | 369.9 ms | 32.12 s |
| `--chunk 100` | 7 | 67.1 MB | 2294.9 ms | 371.7 ms | 32.12 s |
| `--chunk 60` | 11 | 53.2 MB | 2524.9 ms | 311.4 ms | 33.38 s |

Chunking trades total synth time and a little audio length for a 3.2x smaller
peak and a 5.8x faster first audio. Boundaries are clean: the local max
difference at each join is <= 0.00046 against a global p99.9 of 0.177.

## Throughput

`cargo run --release --example roofline`.

Sustained SGEMM at 1024^3 measures **73-99 GFLOP/s** depending on thermal state.
The observed clock under load spans **4.37-4.77 GHz**. Percentages of a
theoretical peak are not quoted here because they move with thermal state; see
`docs/PERFORMANCE_NOTES.md`.

Per-op, 52-token input:

| op | ms | n | GFLOP | GFLOP/s |
|---|---:|---:|---:|---:|
| Conv | 73.2 | 49 | 3.618 | 49.4 |
| MatMul | 27.7 | 67 | 0.932 | 33.7 |
| ConvTranspose | 9.2 | 7 | 0.448 | 48.7 |
| LSTM | 13.6 | 6 | 0.193 | 14.2 |
| **total (flop ops)** | **125.6** | | **5.191** | **41.3** |

The rest of the runtime is memory-bound: `SineExcitation` 21.8 ms, `Add`
21.6 ms, `Transpose` 13.5 ms, `Mul` 9.1 ms.

## Where the time goes

Wall-time accounting for the 52-token input, single thread. One timer per phase,
so the categories sum to the measured wall within 5%.

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
34,158 page faults.

**Measurement trap:** timing every node with `Instant::now()` costs ~7 us per
node, which is 14-25 ms across 2521 nodes. That overhead is distributed by node
count, so it inflates ops with many tiny nodes (`Tanh` 192x, `Pow` 64x, `Pad`
33x). The accounting harness times only ops above a cost threshold and reports
the untimed remainder separately.

## Op-by-op against ONNX Runtime

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

## Reproducing

```bash
# head to head (needs the Python baseline venv)
python scripts/verify/benchmark.py

# fresh-process memory, 5 runs each
python scripts/bench/rss_fresh.py 5

# p50/p95 over 20 runs after 3 warmups
python scripts/bench/bench_stats.py

# interleaved A/B between two commits, ratio only
python scripts/bench/ab_bench.py --a-commit <old> --b-commit <new> --rounds 20

# roofline calibration
cd runtime && cargo run --release --example roofline

# per-op GFLOP/s against the measured peak
python scripts/bench/gflops.py --input mid

# ORT per-op profile
python scripts/verify/ort_profile.py
```

Close background applications before benchmarking. The A/B harness prints the
resolved commit and binary size for each side, because an earlier version
silently compared a binary against itself.
