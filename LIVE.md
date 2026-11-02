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
| `--chunk N` | 80 | Max tokens per synthesis chunk. Smaller = lower first-audio latency, more encoder overhead. |
| `--device NAME` | default | Audio output device name (see `cpal` device list). |

## How It Works

1. **Load once** — Model + G2P loaded at startup (~12 ms)
2. **Chunk at boundaries** — Splits input at sentence (`.`) and comma (`,`) boundaries, max ~80 tokens per chunk
3. **Overlapped synthesis + playback** — Chunk N plays while Chunk N+1 synthesizes via lock-free ring buffer
4. **Auto-resample & interleave** — 24 kHz mono → device rate (e.g., 48 kHz stereo), no external tools
5. **Crossfade** — 10 ms crossfade between chunks to avoid clicks

## Metrics (per line)

```
[live] line 1: first audio at 147 ms (chunk 1/1)
[live]   chunk 1/1: 15 tokens -> 136800 frames (1.43s) in 147 ms
[live] line 1 done: g2p=4.8ms synth=147ms audio=1.43s RTF=9.7x rss=54MB peak=64MB
```

- **first audio** — Time from Enter to first sample sent to device
- **synth** — Total synthesis time for all chunks
- **audio** — Generated audio duration
- **RTF** — Real-time factor (audio_duration / synth_time). Higher = faster
- **rss/peak** — Resident memory during/after synthesis

## Example Output

```
[live] model loaded in 12.3 ms (nodes=2521, inits=515)
[live] warm-up done
[live] rss after warmup: 25.4 MB (peak: 32.2 MB)
[live] audio device: Speakers (Realtek(R) Audio)
[live] audio config: SupportedStreamConfig { channels: 2, sample_rate: 48000 } (resample=true, interleave=true)
[live] ready. Type sentences and press Enter. Ctrl+C to quit.

> Hello world.
[live] line 1: first audio at 147 ms (chunk 1/1)
[live]   chunk 1/1: 15 tokens -> 136800 frames (1.43s) in 147 ms
[live] line 1 done: g2p=4.8ms synth=147ms audio=1.43s RTF=9.7x rss=54MB peak=64MB

> This is a longer sentence. It gets split into chunks.
[live] line 2: first audio at 2.9 s (chunk 1/2)
[live]   chunk 1/2: 50 tokens -> 316800 frames (3.30s) in 264 ms
[live]   chunk 2/2: 62 tokens -> 372000 frames (3.88s) in 301 ms
[live] line 2 done: g2p=6.5ms synth=565ms audio=7.18s RTF=12.7x rss=56MB peak=81MB
```

## Architecture

```
stdin → G2P → token IDs → chunk at punctuation
                              ↓
                    ┌─────────┴─────────┐
                    │  Synth thread     │  (one chunk at a time)
                    │  tinytts::Exec    │
                    └─────────┬─────────┘
                              │ lock-free ring buffer (48k frames)
                              ▼
                    ┌─────────┴─────────┐
                    │  Playback thread  │  (cpal callback)
                    │  resample + interleave
                    └─────────┬─────────┘
                              ▼
                       Audio device
```

- **Zero dependencies in core runtime** — `tinytts` library has no deps
- **Live binary adds** — `cpal` (audio), `winapi` (RSS on Windows), `tinytts-g2p` (text→tokens)
- **Memory** — ~28 MB loaded, ~55 MB during synthesis, peaks ~80 MB for 10s audio
- **Speed** — RTF 9–13x (synthesis 9–13x faster than real-time playback)

## Performance Tips

- Use `--chunk 40`–`60` for lower first-audio latency on long inputs
- Pin synth thread: `start /affinity 2 tinytts-live.exe ...` (Windows)
- Close browser/electron apps before benchmarking