"""Generate and play a sentence, reporting latency breakdown."""

import os
import time

import soundfile as sf

from paradee import SAMPLE_RATE, Paradee

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLES = os.path.join(ROOT, "samples")

TEXT = (
    "But the popup only says, requires approval. Approve to run it, "
    "or deny to keep planning. How will the user know which commands "
    "and work are we talking about?"
)

OUT = os.path.join(SAMPLES, "approval.wav")


def main() -> None:
    tts = Paradee(threads=4)

    t0 = time.perf_counter()
    phonemes = tts.phonemize(TEXT)
    t_g2p = time.perf_counter() - t0

    t0 = time.perf_counter()
    audio = tts.generate_from_phonemes(phonemes)
    t_onnx = time.perf_counter() - t0

    t0 = time.perf_counter()
    sf.write(OUT, audio, SAMPLE_RATE)
    t_write = time.perf_counter() - t0

    duration = len(audio) / SAMPLE_RATE
    total = t_g2p + t_onnx + t_write

    print(f"text chars      : {len(TEXT)}")
    print(f"phoneme chars   : {len(phonemes)}")
    print(f"audio duration  : {duration:.2f} s")
    print()
    print(f"g2p (misaki)    : {t_g2p * 1000:8.1f} ms  ({t_g2p / total * 100:5.1f}%)")
    print(f"onnx inference  : {t_onnx * 1000:8.1f} ms  ({t_onnx / total * 100:5.1f}%)")
    print(f"wav write       : {t_write * 1000:8.1f} ms  ({t_write / total * 100:5.1f}%)")
    print(f"total           : {total * 1000:8.1f} ms")
    print()
    print(f"time to first audio (g2p + onnx): {(t_g2p + t_onnx) * 1000:.1f} ms")
    print(f"real-time factor                : {duration / total:.1f}x")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
