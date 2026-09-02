"""
Phase 2: turn raw_data.csv into labeled, length-truncated train/val/test CSVs.

The full promoter window fetched in fetch_promoter_data.py is -1500/+500 bp
around the TSS (2000 bp, TSS at offset given by the `tss_offset` column).
For a requested SEQ_LEN we take a centered substring using a fixed 3:1
upstream:downstream ratio, so shorter windows stay TSS-centered consistently
with the full window.
"""
import argparse
import numpy as np
import pandas as pd


def truncate(seq, tss_offset, seq_len):
    upstream = int(round(seq_len * 0.75))
    downstream = seq_len - upstream
    start = tss_offset - upstream
    end = tss_offset + downstream
    if start < 0 or end > len(seq):
        return None
    return seq[start:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1000)
    ap.add_argument("--input", default="data/raw_data.csv")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["promoter_seq"] = df.apply(
        lambda r: truncate(r["promoter_seq"], int(r["tss_offset"]), args.seq_len), axis=1
    )
    df = df.dropna(subset=["promoter_seq"])
    df = df[df["promoter_seq"].str.match("^[ACGT]+$")]

    df["log_expression"] = np.log2(df["expression_value"] + 1)
    threshold = df["log_expression"].median()
    df["label"] = (df["log_expression"] > threshold).astype(int)

    train = df.sample(frac=0.8, random_state=42)
    rest = df.drop(train.index)
    val = rest.sample(frac=0.5, random_state=42)
    test = rest.drop(val.index)

    suffix = f"_{args.seq_len}bp"
    train.to_csv(f"{args.outdir}/train{suffix}.csv", index=False)
    val.to_csv(f"{args.outdir}/val{suffix}.csv", index=False)
    test.to_csv(f"{args.outdir}/test{suffix}.csv", index=False)

    print(f"seq_len={args.seq_len}: train={len(train)} val={len(val)} test={len(test)} "
          f"class_balance={df['label'].mean():.3f} threshold(log2 TPM+1)={threshold:.3f}")


if __name__ == "__main__":
    main()
