"""Generate a varied test set to evaluate output quality.

Writes wavs to samples/eval/ and prints a manifest.
"""

import os
import time

import soundfile as sf

from paradee import SAMPLE_RATE, Paradee

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "samples", "eval")

CASES = [
    ("01_plain", "The quick brown fox jumps over the lazy dog."),
    ("02_long_sentence", "Although the committee had reviewed the proposal twice, it remained unclear whether the funding would be approved before the end of the fiscal year."),
    ("03_numbers", "The meeting is on March 3rd, 2026, at 4:30 PM, and the budget is 1,250,000 dollars."),
    ("04_abbrev", "Dr. Smith from the U.S. Dept. of Energy met Mr. Jones at 5 p.m. on NASA property."),
    ("05_questions", "Are you sure? Really? What time does it start? Why would anyone do that?"),
    ("06_exclaim", "Stop! That is incredible! I cannot believe it worked!"),
    ("07_commas", "Well, honestly, I think, if you ask me, that we should probably wait."),
    ("08_alliteration", "She sells seashells by the seashore, and Peter Piper picked a peck of pickled peppers."),
    ("09_technical", "The API returns a JSON payload with an HTTP 404 status code when the endpoint is unreachable."),
    ("10_emotion", "I am so sorry for your loss. Please let me know if there is anything I can do."),
    ("11_quotes", 'He said, "I will be there at noon," and then he left.'),
    ("12_homograph", "I read the book yesterday, and I will read it again tomorrow."),
    ("13_rare_words", "The ubiquitous paradigm of epistemological hegemony is frankly obfuscatory."),
    ("14_short", "Yes."),
    ("15_very_short", "No."),
]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    tts = Paradee(threads=4)

    print(f"{'name':<18} {'chars':>5} {'audio(s)':>9} {'wall(ms)':>9} {'RTF':>7}")
    for name, text in CASES:
        t0 = time.perf_counter()
        audio = tts(text)
        elapsed = time.perf_counter() - t0
        path = os.path.join(OUT, f"{name}.wav")
        sf.write(path, audio, SAMPLE_RATE)
        dur = len(audio) / SAMPLE_RATE
        print(f"{name:<18} {len(text):>5} {dur:>9.2f} {elapsed * 1000:>9.0f} {dur / elapsed:>6.1f}x")

    print(f"\nwrote {len(CASES)} files to {OUT}")


if __name__ == "__main__":
    main()
