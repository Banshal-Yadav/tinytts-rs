"""Generate timestamped audio samples with the Rust runtime.

Writes to audio_out/ (gitignored). Each file is named
  <kind>_<n>_<tokens>tok_<timestamp>.wav
and a sidecar .txt records the text, timing and RSS.

Usage:
    python scripts/user/gen_audio.py                 # 5 long + 5 medium
    python scripts/user/gen_audio.py --long 3 --mid 2
    python scripts/user/gen_audio.py --text "custom sentence"
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import datetime
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXE = os.environ.get("CARGO_TARGET_DIR", os.path.join(ROOT, "runtime", "target")) \
    + r"\release\tinytts.exe"
MODEL = os.path.join(ROOT, "paradee_int8.onnx")
OUT = os.path.join(ROOT, "audio_out")

# Medium: one sentence each. Long: a paragraph each.
MEDIUM_TEXTS = [
    "Hello, this is a test of the speech synthesis system.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Speech synthesis has come a long way in the last few years.",
    "A small model can still sound clear if it is trained well.",
    "Please confirm that the audio sounds natural and free of clicks.",
]

LONG_TEXTS = [
    open(os.path.join(ROOT, "text", "long_text.txt"), encoding="utf-8").read().strip(),
    "The history of speech synthesis is a long and surprisingly stubborn one. "
    "For most of the twentieth century, the best a machine could do was imitate a "
    "human voice in the crudest possible way, buzzing through a bank of analog "
    "filters and hoping the listener would fill in the gaps. Neural networks "
    "changed the rules entirely, and by the early twenty twenties the best systems "
    "were nearly indistinguishable from a person reading aloud.",
    "A voice that runs anywhere is a voice that can live inside a browser tab, a "
    "phone, a cheap embedded board, or a program you wrote on a rainy afternoon. "
    "It will not win a contest against a giant model trained on a thousand hours "
    "of studio recordings, but it does not need to. It needs to be small, fast, "
    "and clear, and on all three counts it delivers.",
    "Latency is the whole game for a conversational agent. If the reply takes a "
    "second to start speaking, the illusion of a conversation breaks. If it takes "
    "half a second, people stop noticing the machine at all. That is the target "
    "this runtime is built for, and every optimisation in it is measured against "
    "that single number.",
    "Memory matters just as much as speed when the target is a laptop, a phone, or "
    "an embedded board. A model that needs a gigabyte of working set cannot run "
    "next to a browser and an editor. A model that needs eighty megabytes can run "
    "almost anywhere, and that is the difference between a demo and a tool.",
]


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_k32 = ctypes.WinDLL("kernel32")
_fn = _k32.K32GetProcessMemoryInfo
_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wt.DWORD]
_fn.restype = wt.BOOL


def peak_mb():
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    _fn(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 1e6


def phonemize(texts):
    """Text -> token ids, using the same G2P the model was trained with."""
    sys.path.insert(0, os.path.join(ROOT, ".venv", "Lib", "site-packages"))
    from paradee.tts import Paradee, MAX_PHONEMES
    p = Paradee(version="1.1", quantized=True, model_path=MODEL)
    out = []
    for t in texts:
        ph = p.phonemize(t)
        ids = [0] + [p.vocab[c] for c in ph if c in p.vocab][:MAX_PHONEMES] + [0]
        out.append(ids)
    return out


def synth(ids, path):
    """Run the runtime once, write the wav, return (synth_ms, peak_rss_mb).

    The runtime writes out.wav relative to its cwd, so run it from runtime/.
    Peak RSS is read from the child process's own report, not this process.
    """
    rt = os.path.join(ROOT, "runtime")
    p = subprocess.run([EXE, MODEL, "--profile"] + [str(i) for i in ids],
                       capture_output=True, text=True, cwd=rt)
    out = p.stdout + p.stderr
    synth_ms = None
    peak = None
    for line in out.splitlines():
        if "synth" in line and ":" in line:
            synth_ms = float(line.split(":")[1].strip().split()[0])
        if line.startswith("rss_peak_mb:"):
            peak = float(line.split(":")[1].strip())
    produced = os.path.join(rt, "out.wav")
    if not os.path.exists(produced):
        raise RuntimeError("runtime did not write out.wav\n" + out[-2000:])
    import shutil
    shutil.move(produced, path)
    return synth_ms, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", type=int, default=5)
    ap.add_argument("--mid", type=int, default=5)
    ap.add_argument("--text", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    jobs = []
    if args.text:
        jobs.append(("custom", args.text))
    else:
        for i in range(min(args.mid, len(MEDIUM_TEXTS))):
            jobs.append(("mid", MEDIUM_TEXTS[i]))
        for i in range(min(args.long, len(LONG_TEXTS))):
            jobs.append(("long", LONG_TEXTS[i]))

    print(f"phonemizing {len(jobs)} texts ...")
    ids_list = phonemize([t for _, t in jobs])

    print(f"{'file':52} {'tokens':>6} {'audio':>8} {'synth':>9} {'peak RSS':>9}")
    for (kind, text), ids in zip(jobs, ids_list):
        n = len(ids)
        name = f"{kind}_{n}tok_{stamp}.wav"
        path = os.path.join(OUT, name)
        synth_ms, pk = synth(ids, path)
        # audio length from the wav header
        import wave
        with wave.open(path) as w:
            frames = w.getnframes()
            rate = w.getframerate()
        audio = frames / rate
        print(f"{name:52} {n:6} {audio:7.2f}s {synth_ms:8.1f}ms {pk:8.1f}MB")
        with open(path[:-4] + ".txt", "w", encoding="utf-8") as f:
            f.write(f"text: {text}\n")
            f.write(f"tokens: {n}\n")
            f.write(f"audio_seconds: {audio:.3f}\n")
            f.write(f"synth_ms: {synth_ms:.1f}\n")
            f.write(f"peak_rss_mb: {pk:.1f}\n")
            f.write(f"generated: {datetime.datetime.now().isoformat()}\n")
            f.write(f"ids: {','.join(str(i) for i in ids)}\n")

    print(f"\nwrote {len(jobs)} files to {OUT}")


if __name__ == "__main__":
    main()
