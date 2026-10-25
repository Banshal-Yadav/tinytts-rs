# Scripts

You only need `user/` to make audio. The other folders are how this runtime was
measured and validated during development.

| folder | what it is for | needs ONNX Runtime |
|---|---|---|
| `user/` | making audio | no |
| `verify/` | correctness gate and head-to-head vs ONNX Runtime | yes |
| `bench/` | performance measurement of this runtime alone | no |
| `lab/` | one-off analysis from the optimization sessions | mixed |

## user/

| script | what it does |
|---|---|
| `setup_model.py` | downloads `paradee_int8.onnx` from Hugging Face |
| `say.py` | text to audio: phonemizes, then runs the Rust binary |
| `speak.py` | same as `say.py`, and plays the result |
| `gen_audio.py` | regenerates the sample clips in the README |
| `stream.py` | streaming demo, writes each chunk as it finishes |
| `stream_play.py` | streaming demo with live playback |

The typical flow:

```bash
python scripts/user/setup_model.py
python scripts/user/say.py "Hello there." -o out.wav
```

`say.py` and `speak.py` need the Python phonemizer (the `paradee` package in
`.venv/`), because text to phoneme ids is the one step this repository does not
implement. The Rust runtime itself needs none of this: it takes phoneme ids
directly. See the README section "Input is phoneme ids, not text".

## verify/

The correctness net. `dump_golden.py` runs ONNX Runtime once with every node
exposed as an output and writes reference tensors (~3.4 GB, not committed).
After that, `cargo test --release` and `gate.py` need no Python and no ORT.

| script | what it does |
|---|---|
| `dump_golden.py` | dumps ORT reference tensors for the differential test |
| `gate.py` | regression gate: PASS/FAIL with the waveform correlation |
| `benchmark.py` | head-to-head timing against ONNX Runtime |
| `ort_profile.py` | ORT per-op profile |
| `ort_long.py` | ORT timing on the 512-token input |
| `ort_rss.py` | ORT memory readout |

## bench/

Measurement of this runtime alone. Absolute milliseconds drift 20%+ between
sessions on a laptop, so speed claims use `ab_bench.py`, which runs two
binaries alternately in one session and reports only the ratio.

| script | what it does |
|---|---|
| `ab_bench.py` | interleaved A/B between two commits, ratio only |
| `bench_stats.py` | p50/p95 over repeated runs |
| `rss_fresh.py` | fresh-process memory numbers |
| `wall_bench.py` | wall-time breakdown across commits |
| `latency.py` | time to first audio |
| `gflops.py` | per-op GFLOP/s against the measured SGEMM peak |
| `mem_probe.py` | live and peak allocation probe |
| `test_stdin_stream.py` | feeds a token stream to `stream_play.py` |

## lab/

One-off analysis from the optimization sessions, kept for the record. Not
needed to build, test or use the runtime.

| script | what it does |
|---|---|
| `chunkcheck.py` | checks the audio quality at chunk joins |
| `chunkdefaults.py` | measured the default chunking settings |
| `chunksweep.py` | swept chunk sizes |
| `deficit_table.py` | per-op time deficit vs ONNX Runtime |
| `analyze_f0.py` | pitch analysis of the output |
| `eval_quality.py` | audio quality evaluation |
