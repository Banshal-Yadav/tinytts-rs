"""Generate misaki reference phonemes + ids for the held-out set.

Same code path as the spike's get_ref.py (paradee.tts.Paradee.phonemize).
Output: heldout_ref.json -> [{"id", "text", "phonemes", "ids"}, ...]
"""
import json
import os
import sys

REPO = r"C:\Users\bansa\OneDrive\Desktop\TTS-8M-model"
os.chdir(REPO)

from paradee.tts import Paradee, MAX_PHONEMES  # noqa: E402

sents = json.load(open(os.path.join(REPO, "heldout.json"), encoding="utf-8"))

p = Paradee(version="1.1", quantized=True, model_path="paradee_int8.onnx")

out = []
for s in sents:
    ph = p.phonemize(s["text"])
    ids = [0] + [p.vocab[c] for c in ph if c in p.vocab][:MAX_PHONEMES] + [0]
    out.append({"id": s["id"], "text": s["text"], "phonemes": ph, "ids": ids})

with open(os.path.join(REPO, "heldout_ref.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"wrote {len(out)} refs")
