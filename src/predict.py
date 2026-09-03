"""
Standalone inference script -- the artifact the Cloud Computing side wraps
in FastAPI/Docker. A checkpoint folder (checkpoints/best_model) plus this
script are all that's required.
"""
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from patch_dnabert2 import ensure_patched
from paths import repo_path

BASE_MODEL_ID = "zhihan1996/DNABERT-2-117M"
MODEL_PATH = repo_path("checkpoints", "best_model")

# Tuned via src/tune_threshold.py: a macro-F1 sweep over P(High) on the
# validation set (never the test set) picked 0.66 over the naive 0.5,
# improving test macro F1 0.67 -> 0.69 and correcting the model's bias
# toward over-predicting High (recall 56%/78% Low/High -> 65%/73%).
# This is specific to checkpoints/best_model -- re-run tune_threshold.py
# and update this constant if the model is ever retrained.
HIGH_THRESHOLD = 0.66


def load():
    ensure_patched()
    # Load the tokenizer from the original base model repo, not from the
    # checkpoint dir. Fine-tuning never changes the tokenizer (vocab/BPE
    # merges are fixed), and save_pretrained() on a newer transformers
    # version can write a tokenizer_config.json (tokenizer_class field)
    # that an older transformers version -- e.g. this repo's local/CPU pin,
    # or the CC team's Docker image -- doesn't recognize and fails to load.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    # low_cpu_mem_usage=False: see comment in train.py -- avoids a meta/cpu
    # device mismatch in DNABERT-2's custom ALiBi tensor construction.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, trust_remote_code=True, low_cpu_mem_usage=False
    )
    model.eval()
    return tokenizer, model


def predict(seq: str, tokenizer=None, model=None, max_length=256):
    if tokenizer is None or model is None:
        tokenizer, model = load()
    inputs = tokenizer(seq, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    # confidence reflects whichever label the threshold actually picked, not
    # just max(probs) -- those diverge once the threshold isn't 0.5 (e.g. at
    # P(High)=0.60 < threshold, the prediction is LOW, so confidence should
    # be P(Low)=0.40, not the larger P(High)=0.60).
    if probs[1].item() >= HIGH_THRESHOLD:
        label, confidence = "HIGH", probs[1].item()
    else:
        label, confidence = "LOW", probs[0].item()
    return {"prediction": f"{label} EXPRESSION", "confidence": round(confidence * 100, 1)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python predict.py <DNA sequence>")
        sys.exit(1)
    print(predict(sys.argv[1]))
