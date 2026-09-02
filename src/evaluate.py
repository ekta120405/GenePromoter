import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset import PromoterDataset
from patch_dnabert2 import ensure_patched

MODEL_ID = "zhihan1996/DNABERT-2-117M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1000)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--checkpoint", default="checkpoints/best_model")
    ap.add_argument("--output", default="results/test_results.txt")
    args = ap.parse_args()

    ensure_patched()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, trust_remote_code=True)
    model.to(device).eval()

    test_ds = PromoterDataset(f"data/test_{args.seq_len}bp.csv", tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds += outputs.logits.argmax(-1).cpu().tolist()
            labels += batch["labels"].cpu().tolist()

    report = classification_report(labels, preds, target_names=["Low", "High"])
    matrix = confusion_matrix(labels, preds)

    print(report)
    print(matrix)

    with open(args.output, "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"test set: data/test_{args.seq_len}bp.csv (n={len(test_ds)})\n\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=true, cols=pred, order=[Low, High]):\n")
        f.write(str(matrix))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
