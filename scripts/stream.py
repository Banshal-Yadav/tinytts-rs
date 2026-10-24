"""Streaming TTS: split text into sentences, synthesize and play each chunk as it is ready.

Reports time-to-first-audio and per-chunk latency.
"""

import argparse
import os
import re
import time

import numpy as np
import soundfile as sf

from paradee import SAMPLE_RATE, Paradee

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")

DEFAULT_TEXT = (
    "But the popup only says, requires approval. Approve to run it, "
    "or deny to keep planning. How will the user know which commands "
    "and work are we talking about?"
)

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_RE.split(text) if p.strip()]
    return parts or [text.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming TTS")
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    parser.add_argument("-f", "--file", help="read text from a UTF-8 file")
    parser.add_argument("-o", "--output", default=os.path.join(SAMPLES, "stream.wav"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--play", action="store_true", help="play each chunk as it is ready")
    args = parser.parse_args()

    text = open(args.file, encoding="utf-8").read() if args.file else args.text
    sentences = split_sentences(text)

    tts = Paradee(threads=args.threads)

    # Warm up so the first chunk is not penalised by lazy init.
    tts.generate_from_phonemes(tts.phonemize("warm up"))

    chunks: list[np.ndarray] = []
    t_start = time.perf_counter()
    t_first_audio = None

    print(f"{'#':>2} {'chars':>5} {'g2p(ms)':>8} {'onnx(ms)':>9} {'audio(s)':>9} {'cum(s)':>7}")
    for i, sentence in enumerate(sentences, 1):
        t0 = time.perf_counter()
        phonemes = tts.phonemize(sentence)
        t_g2p = time.perf_counter() - t0

        t0 = time.perf_counter()
        audio = tts.generate_from_phonemes(phonemes, speed=args.speed)
        t_onnx = time.perf_counter() - t0

        if t_first_audio is None:
            t_first_audio = time.perf_counter() - t_start

        chunks.append(audio)
        cum = time.perf_counter() - t_start
        print(
            f"{i:>2} {len(sentence):>5} {t_g2p * 1000:>8.1f} {t_onnx * 1000:>9.1f} "
            f"{len(audio) / SAMPLE_RATE:>9.2f} {cum:>7.2f}"
        )

        if args.play:
            sf.write(f"_chunk_{i}.wav", audio, SAMPLE_RATE)

    full = np.concatenate(chunks)
    sf.write(args.output, full, SAMPLE_RATE)

    total = time.perf_counter() - t_start
    duration = len(full) / SAMPLE_RATE
    print()
    print(f"chunks          : {len(chunks)}")
    print(f"audio duration  : {duration:.2f} s")
    print(f"time to 1st audio: {t_first_audio * 1000:.1f} ms")
    print(f"total wall      : {total * 1000:.1f} ms")
    print(f"real-time factor: {duration / total:.1f}x")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
