"""
Build raw_data.csv: (gene_id, gene_symbol, promoter_seq, expression_value) pairs.

Sources (both public, no login required):
  - Gene coordinates: Ensembl BioMart (GRCh38), data/gene_coords.tsv
  - Expression: GTEx v10 gene median TPM by tissue, data/GTEx_gene_median_tpm.gct

Promoter sequences are fetched from the Ensembl REST /sequence/region endpoint,
batched 50 regions per request. For every gene we pull a fixed -1500/+500 bp
window around the TSS (2000 bp total, TSS at offset 1500) regardless of strand
-- Ensembl reverse-complements minus-strand regions for us, so the saved
sequence is always given 5'->3' in the gene's own orientation. Phase 2
preprocessing can then take a centered substring of any shorter length
without needing to re-fetch anything.
"""
import argparse
import json
import os
import time
import sys
import requests
import pandas as pd

CACHE_PATH = "data/_seq_cache.json"

UPSTREAM = 1500
DOWNSTREAM = 500
TISSUE = "Liver"
BATCH_SIZE = 50
STD_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y"}
REST_URL = "https://rest.ensembl.org/sequence/region/human"


def load_gene_coords(path="data/gene_coords.tsv"):
    cols = ["ensembl_gene_id", "chrom", "start", "end", "strand", "symbol", "biotype"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols)
    df = df[df["chrom"].isin(STD_CHROMS) & (df["biotype"] == "protein_coding")]
    df = df.drop_duplicates(subset="ensembl_gene_id")
    return df


def load_gtex_tpm(path="data/GTEx_gene_median_tpm.gct", tissue=TISSUE):
    df = pd.read_csv(path, sep="\t", skiprows=2)
    df["ensembl_gene_id"] = df["Name"].str.split(".").str[0]
    df = df[["ensembl_gene_id", tissue]].rename(columns={tissue: "expression_value"})
    # A handful of genes (e.g. pseudoautosomal-region genes) appear twice with
    # conflicting TPM values after stripping the Ensembl version suffix. Drop
    # both copies rather than arbitrarily picking one -- keeping either could
    # silently duplicate the same gene/promoter into both train and test.
    return df.drop_duplicates(subset="ensembl_gene_id", keep=False)


def region_string(row):
    if row["strand"] == 1:
        tss = row["start"]
        g_start, g_end = tss - UPSTREAM, tss + DOWNSTREAM - 1
        strand = 1
    else:
        tss = row["end"]
        g_start, g_end = tss - DOWNSTREAM + 1, tss + UPSTREAM
        strand = -1
    return f"{row['chrom']}:{g_start}..{g_end}:{strand}"


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def fetch_sequences(regions, session, out=None):
    out = {} if out is None else out
    for i in range(0, len(regions), BATCH_SIZE):
        batch = [r for r in regions[i:i + BATCH_SIZE] if r not in out]
        if not batch:
            continue
        for attempt in range(8):
            try:
                resp = session.post(
                    REST_URL,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    json={"regions": batch},
                    timeout=30,
                )
            except requests.exceptions.RequestException:
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code in RETRYABLE_STATUS:
                wait = float(resp.headers.get("Retry-After", min(2 ** attempt, 30)))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            for item in resp.json():
                out[item["query"]] = item["seq"]
            break
        else:
            raise RuntimeError(f"Failed batch starting at {i} after repeated retries")
        time.sleep(0.1)
        done = min(i + BATCH_SIZE, len(regions))
        print(f"  fetched {done}/{len(regions)} regions", file=sys.stderr)
        if (i // BATCH_SIZE) % 10 == 0:
            with open(CACHE_PATH, "w") as f:
                json.dump(out, f)
    with open(CACHE_PATH, "w") as f:
        json.dump(out, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-genes", type=int, default=-1,
                     help="cap the number of genes fetched (random sample); -1 for all matched genes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    coords = load_gene_coords()
    tpm = load_gtex_tpm()
    df = coords.merge(tpm, on="ensembl_gene_id", how="inner")
    df = df.dropna(subset=["expression_value"])
    print(f"{len(df)} protein-coding genes with {TISSUE} expression values", file=sys.stderr)

    if args.max_genes > 0 and args.max_genes < len(df):
        df = df.sample(n=args.max_genes, random_state=args.seed).reset_index(drop=True)
        print(f"Subsampled to {len(df)} genes", file=sys.stderr)

    df["region"] = df.apply(region_string, axis=1)
    session = requests.Session()
    cached = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cached = json.load(f)
        print(f"Resuming from cache: {len(cached)} regions already fetched", file=sys.stderr)
    seqs = fetch_sequences(df["region"].tolist(), session, out=cached)
    df["promoter_seq"] = df["region"].map(seqs)
    df = df.dropna(subset=["promoter_seq"])
    df = df[df["promoter_seq"].str.len() == (UPSTREAM + DOWNSTREAM)]

    out = df[["ensembl_gene_id", "symbol", "chrom", "strand", "promoter_seq", "expression_value"]]
    out = out.rename(columns={"ensembl_gene_id": "gene_id", "symbol": "gene_symbol"})
    out["tss_offset"] = UPSTREAM
    out["tissue"] = TISSUE
    out.to_csv("data/raw_data.csv", index=False)
    print(f"Wrote data/raw_data.csv with {len(out)} genes", file=sys.stderr)


if __name__ == "__main__":
    main()
