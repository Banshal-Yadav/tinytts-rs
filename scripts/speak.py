"""Generate speech with the model.

Usage:
    .venv\\Scripts\\python.exe scripts\\speak.py "Text to speak" [-o out.wav] [--speed 1.0] [--threads 4]
    .venv\\Scripts\\python.exe scripts\\speak.py --file text\\long_text.txt -o samples\\long.wav
"""

import argparse
import os
import time

import soundfile as sf

from paradee import SAMPLE_RATE, Paradee

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="text-to-speech")
    parser.add_argument("text", nargs="?", help="text to speak")
    parser.add_argument("-f", "--file", help="read text from a UTF-8 file instead")
    parser.add_argument("-o", "--output", default=os.path.join(SAMPLES, "output.wav"), help="output wav path")
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = normal speed")
    parser.add_argument("--threads", type=int, default=4, help="onnxruntime intra-op threads")
    args = parser.parse_args()

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        parser.error("provide text or --file")

    tts = Paradee(threads=args.threads)
    t0 = time.perf_counter()
    audio = tts(text, speed=args.speed)
    elapsed = time.perf_counter() - t0

    sf.write(args.output, audio, SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    print(
        f"wrote {args.output} ({duration:.2f} s audio in {elapsed:.2f} s, "
        f"{duration / elapsed:.1f}x real time, {args.threads} threads)"
    )


if __name__ == "__main__":
    main()
