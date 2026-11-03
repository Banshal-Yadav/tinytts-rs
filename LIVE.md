# tinytts-live — Streaming TTS CLI

Real-time text-to-speech in the terminal. Loads model once, streams audio directly to speakers.

## Build

```bash
cd runtime
cargo build --release --bin tinytts-live
```

## Run

```bash
# Basic usage (model path is first argument)
./target/release/tinytts-live.exe ../paradee_int8.onnx

# With options
./target/release/tinytts-live.exe ../paradee_int8.onnx --chunk 60 --device "Speakers"
```

Then type sentences and press Enter. Audio plays immediately. Ctrl+C to quit.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk N` | 40 | Max content tokens per synthesis chunk. Smaller = lower first-audio latency and shorter CPU bursts. Clamped to 510. |
| `--device NAME` | default | Audio output device name (see `cpal` device list). |
| `--speed F` | 1.0 | Feeds the model's `speed` input. 2.0 ≈ half the audio duration (real fast speech, not a pitch shift). |
| `--wav PATH` | off | Also write the rendered 24 kHz mono line to `PATH`. |
| `--rec PATH` | off | Record the default input device while the line plays, as a 16-bit mono WAV at the device rate. For checking what the speakers actually produced. |

**Input limit: 510 content tokens per chunk** (512-slot embedding table, BOS/EOS
take two). Longer input is split automatically; a single call past 512 tokens
reads past the table and panics.

## How It Works

1. **Load once** — Model + G2P loaded at startup (~12 ms)
2. **Chunk at boundaries** — sentence end (`.!?`) → clause break (`;:,`) → hard
   cut, capped at 40 content tokens, runt fragments < 10 tokens merged back
3. **Render the line, then play** — all chunks are synthesized into one 24 kHz
   mono buffer before the device is opened. Chunk N's tail is held back and
   overlap-added into chunk N+1's head (10 ms).
4. **Auto-resample & interleave** — the whole line, once: 24 kHz mono → device
   rate (e.g. 48 kHz stereo), no external tools
5. **Play from a complete buffer** — the cpal callback copies from that finished
   buffer through an atomic cursor, so it cannot run out of data. The callback
   also raises itself to `THREAD_PRIORITY_TIME_CRITICAL` on its first call, so
   it is not competing with the synthesis work for CPU.

> Step 3 used to be a producer/consumer ring: the next chunk was synthesized
> while the current one played. It underran on slower sessions (907 starved
> callbacks in one measured run ≈ 4.5 s of silence inserted mid-sentence).
> Pre-rendering removes that race at the cost of latency: the first word now
> waits for the whole line (≈150 ms for one sentence, ≈3 s for a 500-token
> paragraph). See `## Diagnostics` for what is still unexplained.

## Metrics (per line)

```
[live]   chunk 1/1: 15 tokens -> 36000 frames (1.43s) in 147.2 ms
[live] line 1: 1 chunks, 1.43s of audio synthesised in 147.2 ms (RTF=9.71x), playing
[live] line 1 done: g2p=4.8ms synth=147.2ms audio=1.43s RTF=9.71x play=1.74s rss=54.0MB peak=64.2MB
```

- **synth** — Total synthesis time for all chunks of the line
- **audio** — Generated audio duration
- **RTF** — Real-time factor (audio_duration / synth_time). Higher = faster
- **play** — Wall time the device took to consume the line. Should be
  `audio + ~0.3 s` (the drain grace); anything less means samples were dropped.
- **block** — Frames per WASAPI block (10 ms here), from what the engine asks for.
- **cb** — How many blocks the engine requested. Should be `play / block`.
- **gap_max** — Worst delay between two consecutive block requests. A value
  larger than a few block durations means the engine went hungry and filled the
  gap with silence, which sounds like the narration stopping.
- **starved** — Blocks requested more than 3 block durations late.
- **rss/peak** — Resident memory during/after synthesis

## Example Output

```
[live] model loaded in 10.4 ms (nodes=2521, inits=515)
[live] warm-up done
[live] rss after warmup: 26.2 MB (peak: 31.4 MB)
[live] audio device: Speakers (Realtek(R) Audio)
[live] audio config: SupportedStreamConfig { channels: 2, sample_rate: SampleRate(48000), sample_format: F32 } (resample=true, interleave=true)
[live] ready. Type sentences and press Enter. Ctrl+C to quit.

hey
[live]   chunk 1/1: 5 tokens -> 12600 frames (0.53s) in 83.5 ms
[live] line 1: 1 chunks, 0.53s of audio synthesised in 84.4 ms (RTF=6.22x), playing
[live] line 1 done: g2p=5.5ms synth=84.4ms audio=0.53s RTF=6.22x play=0.84s rss=54.0MB peak=60.9MB
```

`--speed 2.0` on a 512-token paragraph:

```
[live] line 2: 16 chunks, 19.88s of audio synthesised in 2132.5 ms (RTF=9.32x), playing
[live] line 2 done: g2p=0.6ms synth=2132.5ms audio=19.88s RTF=9.32x play=20.18s rss=65.9MB peak=66.4MB
```

## Architecture

```
stdin → G2P → token IDs → chunk at punctuation
                              ↓
                    ┌─────────┴─────────┐
                    │  Synth (main)     │  one chunk at a time
                    │  tinytts::Exec    │  → 24 kHz mono, overlap-added
                    └─────────┬─────────┘
                              │ whole line, resampled + interleaved once
                              ▼
                    ┌─────────┴─────────┐
                    │  Complete PCM buf │  Arc<Vec<f32>>, read-only
                    └─────────┬─────────┘
                              │ atomic cursor copy (cannot starve)
                              ▼
                    ┌─────────┴─────────┐
                    │  cpal callback    │
                    └─────────┬─────────┘
                              ▼
                       Audio device
```

- **Zero dependencies in core runtime** — `tinytts` library has no deps
- **Live binary adds** — `cpal` (audio), `winapi` (RSS on Windows), `tinytts-g2p` (text→tokens)
- **Memory** — ~55 MB loaded, peaks ~66 MB for 20 s of audio (the line's PCM
  buffer is held whole: 4 bytes × channels × device rate × duration)
- **Speed** — RTF 9–13x (synthesis 9–13x faster than real-time playback)

## Performance Tips

- `--chunk 40`–`60` keeps per-chunk bursts short; below ~20 the BOS/EOS padding
  per chunk starts to dominate
- Long lines hold the whole PCM buffer in RAM (~14 MB for 36 s at 48 kHz stereo)
- Pin the process: `start /affinity 2 tinytts-live.exe ...` (Windows)
- Close browser/electron apps before benchmarking

## Diagnostics

If the narration ever sounds like it stops before the line ends, these read the
problem from three different places:

| Question | How to answer it |
|---|---|
| Is the synthesized audio itself broken? | `--wav line.wav`, then inspect the WAV. The render is deterministic, so the file is what went to the device. |
| Did the device *ask* for the samples on time? | `gap_max` and `starved` in the done line. `starved` > 0 means the engine went hungry. |
| Did the speakers actually make the sound? | `--rec room.wav` during playback, then compare 10 ms RMS envelopes of the recording against the `--wav` render. A flat region in the mic during speech means the audio died after it left the process. |

Last measured on the development laptop: `block=10.0ms cb=2017 gap_max=11.4ms
starved=0 play=20.18s` for a 19.88 s line, and a room-mic envelope correlation
of +0.79…+0.83 with no silent second, over six consecutive runs.