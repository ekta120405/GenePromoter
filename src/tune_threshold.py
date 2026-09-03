"""
Post-hoc decision-threshold tuning on an already-trained checkpoint -- no
retraining. The baseline model (argmax / implicit 0.5 threshold) is biased
toward predicting High (78% recall on High vs 56% on Low on the test set).
This script sweeps the threshold on P(High) to maximize macro F1 on the
VALIDATION set (never the test set, to avoid tuning on what should stay
held-out), then reports test-set metrics at both the default and tuned
threshold for comparison.
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset import PromoterDataset
from patch_dnabert2 import ensure_patched
from paths import repo_path

BASE_MODEL_ID = "zhihan1996/DNABERT-2-117M"


def get_probs(model, loader, device):
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            p = torch.softmax(logits, dim=-1)[:, 1]  # P(High)
            probs += p.cpu().tolist()
            labels += batch["labels"].cpu().tolist()
    return np.array(probs), np.array(labels)


def sweep_best_threshold(probs, labels):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 2), best_f1


def report_at_threshold(probs, labels, threshold, title):
    preds = (probs >= threshold).astype(int)
    report = classification_report(labels, preds, target_names=["Low", "High"])
    matrix = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    return (
        f"=== {title} (threshold={threshold:.2f}) ===\n"
        f"accuracy: {acc:.4f}\n\n{report}\n"
        f"Confusion matrix (rows=true, cols=pred, order=[Low, High]):\n{matrix}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1000)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--checkpoint", default="checkpoints/best_model")
    ap.add_argument("--output", default="results/threshold_tuning.txt")
    args = ap.parse_args()

    ensure_patched()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    checkpoint = repo_path(args.checkpoint)
    output_path = repo_path(args.output)

    # Tokenizer from the base model repo, not the checkpoint -- see predict.py
    # for why (cross-transformers-version tokenizer_class incompatibility).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, trust_remote_code=True, low_cpu_mem_usage=False
    )
    model.to(device)

    suffix = f"_{args.seq_len}bp"
    val_ds = PromoterDataset(repo_path("data", f"val{suffix}.csv"), tokenizer, max_length=args.max_length)
    test_ds = PromoterDataset(repo_path("data", f"test{suffix}.csv"), tokenizer, max_length=args.max_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    print(f"Scoring validation set (n={len(val_ds)}) to select threshold...")
    val_probs, val_labels = get_probs(model, val_loader, device)

    default_val_f1 = f1_score(val_labels, (val_probs >= 0.5).astype(int), average="macro")
    best_t, best_val_f1 = sweep_best_threshold(val_probs, val_labels)
    print(f"  default threshold 0.5: val macro F1={default_val_f1:.4f}")
    print(f"  tuned threshold {best_t:.2f}: val macro F1={best_val_f1:.4f}")

    print(f"Scoring test set (n={len(test_ds)})...")
    test_probs, test_labels = get_probs(model, test_loader, device)

    report_default = report_at_threshold(test_probs, test_labels, 0.5, "TEST -- default threshold")
    report_tuned = report_at_threshold(test_probs, test_labels, best_t, "TEST -- tuned threshold")

    full_report = (
        f"checkpoint: {checkpoint}\n"
        f"threshold selected on validation set (n={len(val_ds)}) by macro-F1 sweep, "
        f"never touching the test set for selection\n"
        f"  default (0.5): val macro F1={default_val_f1:.4f}\n"
        f"  tuned ({best_t:.2f}): val macro F1={best_val_f1:.4f}\n\n"
        + report_default + "\n\n" + report_tuned
    )
    print("\n" + full_report)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(full_report + "\n")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
