"""Phase 3 sanity check: how many BPE tokens do our promoter sequences actually need?"""
import sys
import pandas as pd
from transformers import AutoTokenizer

from patch_dnabert2 import ensure_patched


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_1000bp.csv"
    ensure_patched()
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    full = pd.read_csv(csv_path)
    df = full.sample(n=min(200, len(full)), random_state=0)
    lengths = df["promoter_seq"].map(lambda s: len(tokenizer(s)["input_ids"]))
    print(f"{csv_path}: n={len(lengths)} min={lengths.min()} mean={lengths.mean():.1f} "
          f"p95={lengths.quantile(0.95):.0f} max={lengths.max()}")


if __name__ == "__main__":
    main()
