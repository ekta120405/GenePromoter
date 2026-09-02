"""
Standalone inference script -- the artifact the Cloud Computing side wraps
in FastAPI/Docker. A checkpoint folder (checkpoints/best_model) plus this
script are all that's required.
"""
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from patch_dnabert2 import ensure_patched

MODEL_PATH = "checkpoints/best_model"


def load():
    ensure_patched()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    return tokenizer, model


def predict(seq: str, tokenizer=None, model=None, max_length=256):
    if tokenizer is None or model is None:
        tokenizer, model = load()
    inputs = tokenizer(seq, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    label = "HIGH" if probs[1] > probs[0] else "LOW"
    confidence = probs.max().item()
    return {"prediction": f"{label} EXPRESSION", "confidence": round(confidence * 100, 1)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python predict.py <DNA sequence>")
        sys.exit(1)
    print(predict(sys.argv[1]))
