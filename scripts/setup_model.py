"""Inspect the ONNX graph and benchmark thread counts."""

import collections
import glob
import os
import time

import numpy as np
import onnx
import onnxruntime as ort

from paradee import Paradee

HUB = os.path.expanduser(r"~\.cache\huggingface\hub")
MODEL_GLOB = os.path.join(HUB, "blobs", "de", "*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONG_TEXT = os.path.join(ROOT, "text", "long_text.txt")


def find_model():
    for path in glob.glob(MODEL_GLOB):
        if path.endswith(".refs"):
            continue
        if os.path.getsize(path) > 1_000_000:
            return path
    raise SystemExit("model blob not found")


def op_histogram(path):
    model = onnx.load(path)
    counts = collections.Counter(n.op_type for n in model.graph.node)
    print(f"graph nodes: {len(model.graph.node)}")
    print(f"initializers: {len(model.graph.initializer)}")
    for op, n in counts.most_common():
        print(f"  {op:24s} {n}")


def benchmark_threads():
    tts = Paradee()
    text = open(LONG_TEXT, encoding="utf-8").read()
    phonemes = tts.phonemize(text)
    ids = [0] + [tts.vocab[c] for c in phonemes if c in tts.vocab][:510] + [0]
    feed_ids = np.array([ids], dtype=np.int64)
    speed = np.array([1.0], dtype=np.float32)

    model_path = find_model()
    print(f"\nchunk tokens: {feed_ids.shape[1]}")
    print(f"{'threads':>8} {'wall(s)':>9} {'audio(s)':>9} {'RTF':>7}")

    for threads in (1, 2, 4, 8):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

        sess.run(None, {"input_ids": feed_ids, "speed": speed})  # warmup
        best = min(
            _timed_run(sess, feed_ids, speed) for _ in range(3)
        )
        audio_s = sess.run(None, {"input_ids": feed_ids, "speed": speed})[0].shape[-1] / 24000
        print(f"{threads:>8} {best:>9.3f} {audio_s:>9.2f} {audio_s / best:>6.1f}x")


def _timed_run(sess, feed_ids, speed):
    t0 = time.perf_counter()
    sess.run(None, {"input_ids": feed_ids, "speed": speed})
    return time.perf_counter() - t0


if __name__ == "__main__":
    op_histogram(find_model())
    benchmark_threads()
