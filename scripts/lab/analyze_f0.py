"""Measure F0 range and speaking rate for generated wavs.

Compares the condolence sentence against the neutral baseline to test the
"no emotion conditioning -> flat delivery" hypothesis objectively.
"""

import glob
import os

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.path.join(ROOT, "samples", "eval")

FRAME = 1024
HOP = 256
F0_MIN = 60.0
F0_MAX = 400.0


def autocorr_f0(x: np.ndarray, sr: int) -> np.ndarray:
    """Frame-wise F0 via autocorrelation. Returns Hz per frame, 0 where unvoiced."""
    n_frames = 1 + (len(x) - FRAME) // HOP
    f0 = np.zeros(n_frames, dtype=np.float32)
    lo = int(sr / F0_MAX)
    hi = int(sr / F0_MIN)

    for i in range(n_frames):
        frame = x[i * HOP : i * HOP + FRAME].astype(np.float64)
        frame = frame - frame.mean()
        energy = np.sqrt(np.mean(frame**2))
        if energy < 0.01:
            continue
        ac = np.correlate(frame, frame, mode="full")[FRAME - 1 :]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lo:hi]
        if len(seg) == 0:
            continue
        peak = int(np.argmax(seg)) + lo
        if ac[peak] > 0.3:
            f0[i] = sr / peak
    return f0


def analyze(path: str) -> dict:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)

    f0 = autocorr_f0(x, sr)
    voiced = f0[f0 > 0]

    duration = len(x) / sr
    rms = float(np.sqrt(np.mean(x**2)))

    if len(voiced) == 0:
        return {"file": os.path.basename(path), "dur": duration, "rms": rms}

    p5, p50, p95 = np.percentile(voiced, [5, 50, 95])
    return {
        "file": os.path.basename(path),
        "dur": duration,
        "rms": rms,
        "f0_med": float(p50),
        "f0_p5": float(p5),
        "f0_p95": float(p95),
        "f0_range": float(p95 - p5),
        "f0_std": float(np.std(voiced)),
        "voiced_pct": 100.0 * len(voiced) / len(f0),
    }


def main() -> None:
    files = sorted(glob.glob(os.path.join(EVAL, "*.wav")))
    if not files:
        raise SystemExit(f"no wavs in {EVAL}")

    rows = [analyze(f) for f in files]

    print(
        f"{'file':<20} {'dur(s)':>7} {'rms':>6} {'F0med':>6} {'F0p5':>6} "
        f"{'F0p95':>6} {'range':>6} {'std':>6} {'voiced%':>8}"
    )
    for r in rows:
        if "f0_med" not in r:
            print(f"{r['file']:<20} {r['dur']:>7.2f} {r['rms']:>6.3f}  (no voiced frames)")
            continue
        print(
            f"{r['file']:<20} {r['dur']:>7.2f} {r['rms']:>6.3f} {r['f0_med']:>6.1f} "
            f"{r['f0_p5']:>6.1f} {r['f0_p95']:>6.1f} {r['f0_range']:>6.1f} "
            f"{r['f0_std']:>6.1f} {r['voiced_pct']:>7.1f}%"
        )

    base = next((r for r in rows if r["file"].startswith("01_")), None)
    emo = next((r for r in rows if r["file"].startswith("10_")), None)
    if base and emo and "f0_range" in base and "f0_range" in emo:
        print("\n--- hypothesis test: 10_emotion vs 01_plain ---")
        print(f"F0 range   : {emo['f0_range']:.1f} Hz vs {base['f0_range']:.1f} Hz "
              f"({emo['f0_range'] / base['f0_range']:.2f}x)")
        print(f"F0 std     : {emo['f0_std']:.1f} Hz vs {base['f0_std']:.1f} Hz "
              f"({emo['f0_std'] / base['f0_std']:.2f}x)")
        print(f"F0 median  : {emo['f0_med']:.1f} Hz vs {base['f0_med']:.1f} Hz "
              f"({emo['f0_med'] - base['f0_med']:+.1f} Hz)")
        print(f"RMS        : {emo['rms']:.3f} vs {base['rms']:.3f} "
              f"({emo['rms'] / base['rms']:.2f}x)")


if __name__ == "__main__":
    main()
