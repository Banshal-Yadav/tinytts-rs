"""Read text from stdin, synthesize it, play it, exit.

A single blocking command: one line of timing goes to stderr so stdout stays
clean for a caller that wants to capture it.

Usage:
    echo "text" | python say.py
    python say.py "text"
    python say.py --file reply.txt --no-play

By default the wav is written to samples/say.wav and overwritten on each run,
so no temp files accumulate. Use --temp for a throwaway file instead.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import soundfile as sf

from paradee import SAMPLE_RATE, Paradee

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(ROOT, "samples", "say.wav")

PLAYERS = [
    ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ("mpv", ["--no-video", "--really-quiet"]),
    ("vlc", ["--play-and-exit", "--intf", "dummy", "--quiet"]),
]


def find_player():
    for name, flags in PLAYERS:
        exe = shutil.which(name)
        if exe:
            return exe, flags
    return None, None


def play(path: str) -> bool:
    exe, flags = find_player()
    if exe:
        subprocess.run([exe, *flags, path], check=False)
        return True
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - fallback to default handler
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Speak text aloud")
    parser.add_argument("text", nargs="?", help="text to speak; omit to read stdin")
    parser.add_argument("-f", "--file", help="read text from a UTF-8 file")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-play", action="store_true", help="synthesize only")
    parser.add_argument("--keep", default=DEFAULT_OUT, help="wav output path (overwritten each run)")
    parser.add_argument("--temp", action="store_true", help="write to a throwaway temp file instead")
    args = parser.parse_args()

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    text = " ".join(text.split())
    if not text:
        print("say.py: no text provided", file=sys.stderr)
        return 2

    tts = Paradee(threads=args.threads)

    t0 = time.perf_counter()
    audio = tts(text, speed=args.speed)
    t_synth = time.perf_counter() - t0

    if args.temp:
        fd, out = tempfile.mkstemp(suffix=".wav", prefix="say_")
        os.close(fd)
    else:
        out = args.keep
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    sf.write(out, audio, SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE

    played = False
    if not args.no_play:
        played = play(out)

    print(
        f"say.py: {duration:.2f}s audio, synth {t_synth * 1000:.0f}ms, "
        f"{'played' if played else 'not played'} -> {out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
