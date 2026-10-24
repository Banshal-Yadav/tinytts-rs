"""True streaming: synthesize sentence chunks and play them back-to-back with no gaps.

A producer thread synthesizes; a consumer thread plays. Playback of chunk N
overlaps synthesis of chunk N+1, so audio starts after the first chunk and
never stops until the text runs out.
"""

import argparse
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import soundfile as sf

from paradee import SAMPLE_RATE, Paradee

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

DEFAULT_TEXT = (
    "This is true streaming. The producer thread synthesizes one sentence at a time "
    "while the consumer thread plays the previous one. Audio starts after the first "
    "chunk and never stops until the text runs out. That is how you get a voice that "
    "keeps up with a model that is still writing."
)


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_RE.split(text) if p.strip()]
    return parts or [text.strip()]


TERMINATOR_RE = re.compile(r"[.!?\u2026]+[\"'\u201d\u2019)]*\s+")


def iter_stdin_sentences(min_chars: int = 12):
    """Yield sentences as they arrive on stdin.

    Flushes on a sentence terminator followed by whitespace, or on a newline.
    A trailing fragment shorter than min_chars is held back so we do not speak
    half a clause; it is flushed at EOF.
    """
    buf = ""
    while True:
        chunk = sys.stdin.read(1)
        if not chunk:
            break
        buf += chunk

        while True:
            m = TERMINATOR_RE.search(buf)
            if not m:
                break
            sentence = buf[: m.end()].strip()
            buf = buf[m.end() :]
            if sentence:
                yield sentence

        if "\n" in buf:
            head, _, tail = buf.partition("\n")
            buf = tail
            if head.strip():
                yield head.strip()

        if len(buf) >= 200:
            cut = buf.rfind(" ")
            if cut > 0:
                yield buf[:cut].strip()
                buf = buf[cut:].lstrip()

    if buf.strip():
        yield buf.strip()


def find_player():
    for name, flags in (
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ("mpv", ["--no-video", "--really-quiet"]),
    ):
        exe = shutil.which(name)
        if exe:
            return exe, flags
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming TTS with overlapped playback")
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    parser.add_argument("-f", "--file", help="read text from a UTF-8 file")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-play", action="store_true", help="synthesize only, no playback")
    parser.add_argument(
        "--stdin-stream",
        action="store_true",
        help="read text incrementally from stdin, speak each sentence as it completes",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=12,
        help="with --stdin-stream, flush a fragment once it reaches this length",
    )
    args = parser.parse_args()

    if args.stdin_stream:
        sentences = None  # produced incrementally
    else:
        text = open(args.file, encoding="utf-8").read() if args.file else args.text
        sentences = split_sentences(text)

    tts = Paradee(threads=args.threads)
    tts.generate_from_phonemes(tts.phonemize("warm up"))  # pay session init up front

    exe, flags = find_player()
    if not args.no_play and exe is None:
        print("stream_play: no ffplay/mpv found, falling back to --no-play", file=sys.stderr)
        args.no_play = True

    q: queue.Queue = queue.Queue()
    stats = {"first_audio": None, "synth_total": 0.0, "chunks": 0}
    t_start = time.perf_counter()

    def producer():
        if args.stdin_stream:
            for i, sentence in enumerate(iter_stdin_sentences(args.min_chars), 1):
                emit(i, sentence)
        else:
            for i, sentence in enumerate(sentences, 1):
                emit(i, sentence)
        q.put(None)

    def emit(i, sentence):
        t0 = time.perf_counter()
        audio = tts.generate_from_phonemes(tts.phonemize(sentence), speed=args.speed)
        dt = time.perf_counter() - t0
        stats["synth_total"] += dt
        stats["chunks"] += 1
        q.put((i, sentence, audio, dt))

    def consumer():
        while True:
            item = q.get()
            if item is None:
                break
            i, sentence, audio, dt = item
            if stats["first_audio"] is None:
                stats["first_audio"] = time.perf_counter() - t_start

            if args.no_play:
                print(
                    f"  chunk {i}: {len(audio) / SAMPLE_RATE:5.2f}s audio, "
                    f"synth {dt * 1000:5.0f}ms  | {sentence[:50]}"
                )
                continue

            fd, path = tempfile.mkstemp(suffix=".wav", prefix="sp_")
            os.close(fd)
            sf.write(path, audio, SAMPLE_RATE)
            subprocess.run([exe, *flags, path], check=False)
            try:
                os.remove(path)
            except OSError:
                pass

    tp = threading.Thread(target=producer, daemon=True)
    tc = threading.Thread(target=consumer, daemon=True)
    tp.start()
    tc.start()
    tp.join()
    tc.join()

    total = time.perf_counter() - t_start
    first = stats["first_audio"]
    print()
    print(f"chunks           : {stats['chunks']}")
    print(f"time to 1st audio: {first * 1000:.0f} ms" if first is not None else "time to 1st audio: n/a")
    print(f"synth total      : {stats['synth_total'] * 1000:.0f} ms")
    print(f"wall total       : {total * 1000:.0f} ms")


if __name__ == "__main__":
    main()
