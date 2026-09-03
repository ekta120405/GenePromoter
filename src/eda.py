"""
Exploratory data analysis on the promoter-expression dataset.
Reads data/raw_data.csv + data/{train,val,test}_1000bp.csv, writes a summary
figure and text report to eda/. Run after fetch_promoter_data.py + preprocess.py.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1000)
    ap.add_argument("--outdir", default="eda")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    raw = pd.read_csv("data/raw_data.csv")
    suffix = f"_{args.seq_len}bp"
    train = pd.read_csv(f"data/train{suffix}.csv")
    val = pd.read_csv(f"data/val{suffix}.csv")
    test = pd.read_csv(f"data/test{suffix}.csv")
    full = pd.concat([train, val, test], ignore_index=True)
    full["gc_content"] = full["promoter_seq"].map(gc_content)

    lines = []
    lines.append(f"raw_data.csv: {len(raw)} genes, {raw.gene_id.duplicated().sum()} duplicate gene_ids")
    lines.append(f"missing gene_symbol: {raw.gene_symbol.isnull().sum()}")
    lines.append(f"zero-TPM genes: {(raw.expression_value == 0).sum()} ({(raw.expression_value == 0).mean():.1%})")
    lines.append(f"raw TPM: min={raw.expression_value.min():.3f} median={raw.expression_value.median():.3f} "
                  f"max={raw.expression_value.max():.1f} mean={raw.expression_value.mean():.2f}")
    lines.append("")
    lines.append(f"train={len(train)} val={len(val)} test={len(test)} "
                  f"(class balance: {train.label.mean():.3f} / {val.label.mean():.3f} / {test.label.mean():.3f})")
    n_overlap = len(set(train.gene_id) & set(val.gene_id)) + len(set(train.gene_id) & set(test.gene_id)) \
        + len(set(val.gene_id) & set(test.gene_id))
    lines.append(f"cross-split gene_id overlap: {n_overlap} (must be 0)")
    lines.append("")
    lines.append(f"GC content: mean={full.gc_content.mean():.3f} std={full.gc_content.std():.3f}")
    lines.append(f"  High-expression mean GC: {full[full.label == 1].gc_content.mean():.3f}")
    lines.append(f"  Low-expression  mean GC: {full[full.label == 0].gc_content.mean():.3f}")
    report = "\n".join(lines)
    print(report)
    with open(f"{args.outdir}/eda_summary.txt", "w") as f:
        f.write(report + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].hist(np.log10(raw.expression_value + 1e-3), bins=50, color="#4C72B0")
    axes[0, 0].set_title("Raw expression: log10(TPM + 1e-3)")
    axes[0, 0].set_xlabel("log10(TPM)")

    axes[0, 1].hist(full.log_expression, bins=50, color="#55A868")
    axes[0, 1].axvline(full.log_expression.median(), color="red", linestyle="--", label="median (split threshold)")
    axes[0, 1].set_title("log2(TPM + 1) with label threshold")
    axes[0, 1].legend()

    splits = ["train", "val", "test"]
    balances = [train.label.mean(), val.label.mean(), test.label.mean()]
    axes[0, 2].bar(splits, balances, color="#C44E52")
    axes[0, 2].axhline(0.5, color="gray", linestyle=":")
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_title("Class balance (fraction High) per split")

    chrom_counts = raw.chrom.value_counts()
    order = [str(i) for i in range(1, 23)] + ["X", "Y"]
    chrom_counts = chrom_counts.reindex([c for c in order if c in chrom_counts.index])
    axes[1, 0].bar(chrom_counts.index, chrom_counts.values, color="#8172B2")
    axes[1, 0].set_title("Genes per chromosome")
    axes[1, 0].tick_params(axis="x", labelrotation=90)

    axes[1, 1].hist(full[full.label == 0].gc_content, bins=40, alpha=0.6, label="Low", color="#C44E52")
    axes[1, 1].hist(full[full.label == 1].gc_content, bins=40, alpha=0.6, label="High", color="#55A868")
    axes[1, 1].set_title(f"GC content by label ({args.seq_len}bp window)")
    axes[1, 1].legend()

    axes[1, 2].scatter(full.gc_content, full.log_expression, s=4, alpha=0.3, color="#4C72B0")
    axes[1, 2].set_title("GC content vs log2(TPM+1)")
    axes[1, 2].set_xlabel("GC content")
    axes[1, 2].set_ylabel("log2(TPM+1)")

    fig.tight_layout()
    fig.savefig(f"{args.outdir}/eda_summary.png", dpi=150)
    print(f"\nSaved {args.outdir}/eda_summary.png and {args.outdir}/eda_summary.txt")


if __name__ == "__main__":
    main()
