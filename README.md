# GenePromoter

Cloud-Enabled Deep Learning for Predicting High and Low Gene Expression from Promoter DNA Sequences

Fine-tunes DNABERT-2 to classify human promoter sequences as High or Low expression. This repo covers the DL side of the project (data -> tokenize -> fine-tune -> evaluate -> checkpoint -> inference script); the Cloud Computing (Docker/Kubernetes) side wraps `src/predict.py` + a saved checkpoint separately.

## Environment

No conda, no NVIDIA GPU on this machine -- training runs on CPU with the system Python (3.12) in a venv.

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate.bat on cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`transformers==4.29.2` (what DNABERT-2's own docs suggest) has no prebuilt wheel for Python 3.12 and needs a Rust compiler to build from source, so this repo pins a modern version (4.40.0) instead. That trades one problem for two others, both patched automatically (see below):

1. **triton import**: DNABERT-2's remote code (`bert_layers.py`) imports a triton-based flash-attention module. `transformers` refuses to load the model at all if `triton` isn't installed, even though the code already falls back to standard attention on `ImportError`. There's no CPU wheel for `triton` on Windows.
2. **`config_class` mismatch**: DNABERT-2's custom `BertModel`/`BertForSequenceClassification` classes don't declare their own `config_class`, so `AutoModel.from_pretrained` derives it from the wrong (built-in) `BertConfig` and raises `ValueError` during auto-registration.

[src/patch_dnabert2.py](src/patch_dnabert2.py) downloads the model repo's files and patches the cached snapshot in place to fix both. It's idempotent and every other script (`train.py`, `evaluate.py`, `predict.py`, `check_tokenization.py`) calls `ensure_patched()` before touching the model, so this isn't a manual one-off step -- it's reproducible on a fresh machine or after clearing the HF cache.

## Dataset

The roadmap's original source (the Xpresso repo) ships as an ~11GB unstructured dump on a university server with no documented file layout -- not practical here. Instead, following the roadmap's own sanctioned fallback ("any GTEx-derived processed dataset"), the dataset is built from two public sources with no login required:

- **Expression labels**: GTEx v10 gene median TPM by tissue (`data/GTEx_gene_median_tpm.gct`, ~59k genes x 68 tissues). Liver is used as the target tissue.
- **Promoter sequences**: gene coordinates from Ensembl BioMart (GRCh38), then a -1500/+500bp TSS-centered window per gene fetched from the Ensembl REST `/sequence/region` endpoint (strand-aware; minus-strand genes are reverse-complemented automatically so every saved sequence is 5'->3' in the gene's own orientation).

[src/fetch_promoter_data.py](src/fetch_promoter_data.py) does the join and fetch (resumable via `data/_seq_cache.json`, retries on transient 5xx/429). By default it fetches every matched protein-coding gene (~19k); pass `--max-genes N` to cap it for a faster run.

```bash
python src/fetch_promoter_data.py --max-genes 6000   # -> data/raw_data.csv
python src/preprocess.py --seq-len 1000               # -> data/{train,val,test}_1000bp.csv
```

`preprocess.py` truncates each gene's fetched 2000bp window to a centered substring of any requested length (fixed 3:1 upstream:downstream ratio, so 500/1000/2000bp variants all stay TSS-centered without re-fetching), labels High/Low by median split on log2(TPM+1), and writes the train/val/test CSVs.

## Pipeline

```bash
python src/check_tokenization.py data/train_1000bp.csv   # verify max_length is sane
python src/train.py --seq-len 1000 --max-train-samples 2000 --epochs 1   # small first run
python src/evaluate.py --seq-len 1000                                    # -> results/test_results.txt
python src/predict.py ACGTACGTACGT...
```

`train.py` defaults to a 2000-example subsample and 1 epoch so a first run stays fast on any machine. Pass `--max-train-samples -1 --epochs 5` for a full-scale run. `--freeze-base` runs the frozen-vs-fine-tuned stretch-goal comparison.

**Known issue on this machine:** a CPU training run of `train.py` segfaults on this Windows/CPU-only box partway through the first forward/backward pass (exit code 139), cause not yet root-caused -- possibly an OpenMP/MKL DLL conflict between torch/numpy/scikit-learn, or a bug in DNABERT-2's custom CPU pad/unpad attention path, which was written and tested primarily for GPU. `check_tokenization.py` (which only runs the tokenizer, not the model) works fine, so the model load itself is not the problem. This hasn't been debugged further because the plan is to train on Colab/GPU instead -- worth revisiting if CPU training is ever needed again.

## Running on Google Colab

`data/raw_data.csv` and `data/{train,val,test}_1000bp.csv` are committed to this repo specifically so Colab doesn't need to re-hit the (slow, rate-limited) Ensembl API -- clone and go straight to training.

```python
!git clone https://github.com/ekta120405/GenePromoter
%cd GenePromoter
!pip install -q transformers==4.40.0 einops accelerate scikit-learn pandas numpy datasets requests
```

Don't reinstall `torch` -- Colab's preinstalled build already matches its CUDA runtime; installing the pinned CPU-era version from `requirements.txt` would replace it. Everything else in `requirements.txt` is safe to install as pinned.

```python
!python src/train.py --seq-len 1000 --max-train-samples -1 --epochs 5   # full-scale, GPU-accelerated
!python src/evaluate.py --seq-len 1000
```

`train.py` already does `device = "cuda" if torch.cuda.is_available() else "cpu"`, so no code changes are needed to pick up Colab's GPU. Runtime -> Change runtime type -> T4 GPU (free tier) is enough for fine-tuning a 117M-param model. After training, download `checkpoints/best_model/` (or copy it to Drive) to bring the fine-tuned weights back for `predict.py` / the CC handoff.
