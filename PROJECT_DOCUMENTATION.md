# GenePromoter -- Full Project Documentation

This document explains the entire DL pipeline end to end: the biology behind the task, every data-processing decision, the model, every compatibility issue hit and how it was fixed, the training/evaluation methodology, and the final results. It's meant to be readable standalone -- by a teammate, a faculty reviewer, or future-you six months from now.

---

## 1. What this project is actually doing

**Goal:** predict whether a human gene is highly or lowly expressed in liver tissue, using only the DNA sequence of its promoter region -- no other biological features, no prior knowledge injected, just raw sequence in and a High/Low label out.

**Why this is a real question:** a gene's promoter (the DNA immediately around where transcription starts) contains binding sites for the proteins that control whether and how much that gene gets transcribed. The hypothesis this project tests is: does a pretrained genomic Transformer (DNABERT-2) contain enough general understanding of DNA structure that, after fine-tuning on a modest labeled dataset (~5,800 genes), it can learn to recognize the sequence patterns that correlate with expression level -- without anyone hand-engineering what those patterns should look like.

**The biological vocabulary used throughout this doc:**
- **Gene expression**: how actively a gene is being transcribed into RNA. Measured here as **TPM** (Transcripts Per Million) -- a standard RNA-seq unit, normalized so it's comparable across genes and samples.
- **Promoter**: the stretch of DNA immediately upstream (and a little downstream) of a gene, where the transcription machinery assembles. This is the region we feed the model.
- **TSS (Transcription Start Site)**: the exact base pair where RNA polymerase begins transcribing. All promoter windows in this project are defined relative to the TSS.
- **bp (base pairs)**: the unit of DNA sequence length. "1000bp" = a 1000-nucleotide stretch.
- **Strand**: DNA is double-stranded; a gene is transcribed from one strand or the other. Sequence orientation matters (a promoter read backwards is biologically meaningless), so every fetched sequence is oriented 5'->3' in the gene's own transcriptional direction, regardless of which physical strand it sits on.

---

## 2. Data: where it comes from and how it was built

There is no single off-the-shelf "promoter sequence + expression level" dataset used here. The roadmap's originally suggested source (the Xpresso repo) turned out to be an ~11GB unstructured dump with no documented file layout, impractical to use. Instead, the dataset is assembled from two independent, public, no-login-required sources, joined by `src/fetch_promoter_data.py`:

- **Expression labels: GTEx v10**, gene median TPM by tissue (`GTEx_gene_median_tpm.gct`, ~59k genes x 68 tissues). **Liver** is the target tissue for this whole project -- expression is tissue-specific, so this label only means "high/low in liver," not universally.
- **Promoter sequences: Ensembl BioMart + REST API** (GRCh38 human genome build). For each gene, a fixed **-1500/+500bp window around the TSS** (2000bp total) is fetched via the `/sequence/region` endpoint, batched 50 regions per request with retry/backoff on rate limits and resumable caching (`data/_seq_cache.json`) since the API is slow.

**Strand handling:** Ensembl reverse-complements minus-strand genes automatically, so every saved sequence is already oriented 5'->3' relative to the gene's own transcription direction -- this matters because sequence motifs (like a TATA box) are directional, and mixing orientations would make half the dataset biologically backwards relative to the other half.

### `raw_data.csv` schema

| Column | Meaning |
|---|---|
| `gene_id` | Ensembl stable gene ID |
| `gene_symbol` | Human-readable gene name (not used by the model -- for humans reading the CSV) |
| `chrom`, `strand` | Genomic location |
| `promoter_seq` | The raw 2000bp DNA string (A/C/G/T), TSS at offset 1500 |
| `expression_value` | GTEx median Liver TPM -- a continuous number, e.g. 0.0 to 25339.7 |
| `tss_offset` | Index into `promoter_seq` where the TSS sits (1500 in the raw file) |
| `tissue` | Always "Liver" here |

No labels exist at this stage -- just sequence + a continuous expression number. **High/Low is invented downstream**, not something GTEx or Ensembl provides.

---

## 3. Preprocessing (`src/preprocess.py`)

Run as: `python src/preprocess.py --seq-len 1000`

1. **Truncate to a centered window.** The full fetched window is 2000bp (-1500/+500). For any requested `--seq-len`, a centered substring is taken using a fixed **3:1 upstream:downstream ratio** (e.g. 1000bp = -750/+250), so 500/1000/2000bp variants are all TSS-centered without needing to re-fetch anything from Ensembl.
2. **Filter to pure ACGT.** Drops any sequence containing ambiguous bases.
3. **Log-transform expression.** `log_expression = log2(expression_value + 1)`. Raw TPM is heavily right-skewed (a small number of genes have extreme values, most sit low), so log2(x+1) compresses that tail into something workable and cleanly maps zero-TPM genes to exactly 0 (no arbitrary epsilon needed).
4. **Label via global median split.** `label = 1 (High) if log_expression > median else 0 (Low)`, where the median is computed **once, across the whole dataset, before splitting** into train/val/test. This is why every split ends up naturally class-balanced without needing explicit stratified sampling.
5. **Split 80/10/10** into train/val/test, randomly, after labeling.

**Important framing:** "High expression" here is a *relative, dataset-defined* label -- "above the median of this specific ~5,800-gene liver sample" -- not a universal biological threshold. A model trained on this label should not be assumed to transfer to a different tissue or a different gene population without re-deriving the threshold from that population's own data.

---

## 4. The data-leakage bug (found and fixed)

Early exploratory checks (not a full formal EDA yet at that point) found: `raw_data.csv` had **2 genes with duplicate `gene_id` entries**, each with **conflicting TPM values** (e.g. `AKAP17A`: 44.36 vs 0.0 TPM). Root cause: `fetch_promoter_data.py`'s `load_gtex_tpm()` didn't dedupe the GTEx TPM table before joining against gene coordinates -- a gene with two TPM rows in GTEx's own file (plausibly pseudoautosomal-region genes, which GTEx sometimes double-lists) fanned out into two rows in `raw_data.csv`.

**Consequence:** because `preprocess.py` splits by row (not by deduplicated gene), one of these duplicated genes (`AKAP17A`) ended up in **both train and test simultaneously**, labeled High in train and Low in test -- the same promoter sequence present in a supposedly held-out test set.

**Fix:** `load_gtex_tpm()` now drops both copies of any gene with conflicting duplicate TPM values (`drop_duplicates(subset="ensembl_gene_id", keep=False)` -- rather than arbitrarily keeping one, since we don't know which value is correct). `raw_data.csv` was regenerated (6000 -> 5996 genes after removing the two ambiguous ones), splits were regenerated, and **cross-split gene_id overlap was verified at exactly 0** across all three splits. This is now checked automatically in `src/eda.py`'s report.

---

## 5. Exploratory Data Analysis (`src/eda.py`)

Run as: `python src/eda.py --seq-len 1000`. Produces `eda/eda_summary.png` (6 plots) and `eda/eda_summary.md` (per-graph inference + conclusion, all numbers computed live from the data, not hand-typed).

**Headline findings:**
- Raw TPM is heavily right-skewed (skewness ~50.8; mean 25.41 >> median 2.315) -- confirms the log-transform is necessary, not optional.
- 766 genes (12.8%) have exactly zero measured TPM in liver -- real biology (tissue-specific genes genuinely off in liver), not missing data.
- Class balance after the median split is close to 50/50 in every split (train 0.497, val 0.487, test 0.533), and the small deviations are within normal sampling noise for each split's size.
- Chromosome representation looks like a genuine genome-wide sample (chr1 has the most genes, chrY the fewest -- matches known biology), confirming no fetch bias.
- **GC content, a trivial one-line sequence statistic, already separates High vs Low classes**: 0.555 vs 0.523 mean GC (Cohen's d = 0.335, small-to-moderate effect), and correlates with log-expression at r=0.147 (r^2=0.022, ~2.2% of variance explained). This is the key validation finding: it proves the classification task is grounded in real, independently-verifiable promoter biology (GC-rich/CpG-island promoters skew toward higher/broader expression) -- while also being far too weak on its own to solve the task, which is the actual argument for using a sequence-aware Transformer instead of a hand-engineered feature classifier.

See `eda/eda_summary.md` for the full per-graph writeup, effect sizes, and stated caveats (tissue-specificity of the label, window-size limitations, GTEx's own cell-heterogeneity blind spot).

---

## 6. The model: DNABERT-2

**What it is:** a Transformer (BERT-family) pretrained on genomic DNA sequences from multiple species using masked-language-modeling -- before this project touches it, it already has a general statistical understanding of DNA structure (recurring motifs, base-transition patterns, etc.), the same way a language BERT understands general English before being fine-tuned for a specific task.

**What's different from a plain BERT / from the original DNABERT:**
- **Tokenization**: DNABERT-2 uses Byte-Pair Encoding (BPE) learned over DNA sequences, producing variable-length tokens -- not fixed-length k-mers like the original DNABERT. A 1000bp sequence tokenizes to roughly 200-230 tokens (verified via `src/check_tokenization.py`: min=176, mean=204.2, p95=217, max=229 on a 200-sample check), comfortably inside the `max_length=256` token budget used throughout.
- **Architecture**: `hidden_size=768, num_hidden_layers=12, num_attention_heads=12, hidden_act=gelu` (confirmed from the model's own `config.json`) -- a standard-sized BERT-base-equivalent, ~117M parameters.
- **ALiBi position encoding**: instead of learned absolute position embeddings, attention bias is computed from relative token distance (ALiBi -- Attention with Linear Biases), precomputed into a tensor at model init and dynamically resized/moved as needed during the forward pass.

**Why fine-tune instead of train from scratch:** ~4,800 labeled training examples is far too little to train a 117M-parameter Transformer from scratch without severe overfitting. Transfer learning -- starting from weights that already understand general DNA structure, then adapting them to this specific classification task -- is what makes the problem tractable at this dataset size.

**Mechanically, what happens on one input:** DNA string -> BPE tokenizer -> token IDs + attention mask -> token embeddings -> 12 Transformer self-attention layers (each token's representation built by attending to every other token, in principle picking up motifs and their spatial relationships) -> pooled sequence representation (via a pooler layer) -> a linear classification head -> 2 logits (Low, High) -> softmax -> probabilities -> thresholded into a final label.

**Input/output contract, precisely:**
- **Input**: a single DNA sequence string, uppercase A/C/G/T only, arbitrary length (the tokenizer truncates/pads to `max_length=256` tokens, corresponding to roughly the first ~1000bp worth of content -- see `dataset.py`'s `PromoterDataset`).
- **Output** (from `predict.py`): `{"prediction": "HIGH EXPRESSION" | "LOW EXPRESSION", "confidence": <0-100>}`.
- **Training-time input** additionally includes the ground-truth `label` (0/1), used only to compute the loss for backpropagation -- never fed into the model as a feature.

---

## 7. Environment compatibility: `src/patch_dnabert2.py`

DNABERT-2's HuggingFace repo (`zhihan1996/DNABERT-2-117M`) ships remote code (`bert_layers.py`, `configuration_bert.py`) written against an old `transformers` version and never updated. Running it against any modern `transformers` install -- whether this repo's pinned local/CPU version or Colab's much newer preinstalled one -- breaks in **four distinct ways**, each hit for real during this project and each now patched automatically:

1. **Triton import.** `bert_layers.py` unconditionally imports a triton-based flash-attention module. `transformers`' `check_imports` statically refuses to load the model at all if `triton` isn't installed -- even though the code already falls back to standard attention on `ImportError`. There's no CPU/non-flash-attention `triton` wheel for Windows. **Fix**: the import is removed outright, forcing the standard-attention fallback unconditionally.
2. **`config_class` mismatch.** None of `BertModel`/`BertForMaskedLM`/`BertForSequenceClassification` in the remote code declare their own `config_class`, so they inherit the *real* `transformers.models.bert.modeling_bert.BertPreTrainedModel`'s built-in `BertConfig`. `AutoModel.from_pretrained` then finds a mismatch against the repo's own custom `BertConfig` and raises `ValueError` during auto-registration. **Fix**: `config_class = BertConfig` (the correct custom class) is injected into each of the three model classes.
3. **Missing `pad_token_id`.** `BertEmbeddings.__init__` reads `config.pad_token_id` directly. Older `transformers` always set this attribute (defaulting to `None`) in `PretrainedConfig.__init__`, so direct access was safe. On the much newer `transformers` version Colab preinstalls, that guarantee doesn't hold for this custom config subclass, and access raises `AttributeError` instead of returning `None`. **Fix**: switched to `getattr(config, "pad_token_id", None)` -- `None` is a perfectly valid `padding_idx` for `nn.Embedding` (means "no padding index"), so no behavior actually changes, just the access pattern.
4. **Meta/cpu device mismatch in ALiBi construction.** `BertEncoder.__init__` eagerly builds an ALiBi bias tensor via `rebuild_alibi_tensor(size=..., device=None)`. Newer `transformers` initializes models inside a `with torch.device("meta"):` context for speed (materializing real weights only after, from the state dict). Modern factory calls like `torch.arange(..., device=None)` obey that ambient context and land on `meta`; the legacy `torch.Tensor(list)` constructor used for the `slopes` tensor does not, and always lands on real `cpu`. Multiplying a `meta` tensor against a `cpu` tensor raises `RuntimeError: Tensor on device meta is not on the expected device cpu!`. **Fix**: `rebuild_alibi_tensor` now resolves `device=None` to `"cpu"` explicitly at the top of the function -- an explicit device argument always overrides the ambient context, restoring the pre-meta-init behavior this code was originally written against.

**Two more subtleties worth recording**, both hit and fixed during actual Colab runs:

- **The patch has to hit two separate caches.** `trust_remote_code=True` loading copies the repo's `.py` files a *second* time, into `~/.cache/huggingface/modules/transformers_modules/...` -- that copy, not the hub snapshot cache, is what's actually imported, and `transformers` does not re-sync it from the hub snapshot on later calls once it exists. `patch_dnabert2.py` patches both locations every time `ensure_patched()` runs.
- **The modules-cache directory name isn't stable across transformers versions.** Some versions name it `DNABERT-2-117M` (as-is), others sanitize hyphens to `DNABERT_hyphen_2_hyphen_117M`. The patch glob is wildcarded (`zhihan1996/*/*/bert_layers.py`) rather than hardcoding either spelling, after a hardcoded guess silently failed to match on Colab and let an unpatched file through undetected.

All of this is idempotent and automatic -- every script that touches the model (`train.py`, `evaluate.py`, `predict.py`, `check_tokenization.py`, `tune_threshold.py`) calls `ensure_patched()` before doing anything else. It is not a manual one-time step.

---

## 8. Training (`src/train.py`)

Run as (full-scale, on a GPU): `python src/train.py --seq-len 1000 --max-train-samples -1 --epochs 5`

**Setup:**
- Model: `AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2, low_cpu_mem_usage=False)` -- `low_cpu_mem_usage=False` avoids yet another meta-device interaction; `num_labels=2` attaches a fresh linear classification head (`classifier.weight`/`classifier.bias`) on top of the pretrained encoder. The pooler layer (`bert.pooler.dense`) is *also* freshly initialized -- it's not part of the pretrained checkpoint either (confirmed `MISSING` in every load report).
- **Optimizer**: AdamW, `lr=3e-5` (default), `weight_decay=0.01` -- the standard, near-default-optimal choice for Transformer fine-tuning.
- **LR schedule**: linear warmup (50 steps) then linear decay to 0, via `get_linear_schedule_with_warmup`.
- **Loss**: cross-entropy (computed internally by the model when `labels` are passed), the correct choice for binary classification.
- **Gradient clipping**: max norm 1.0, standard practice against exploding gradients.
- **Batch size**: 8.
- **Early stopping**: patience=2 (stop if 2 consecutive epochs don't beat the best validation F1 so far); the checkpoint is only overwritten when validation F1 improves, so the saved model is always the best-validation-performing one, not simply the last epoch trained.

**Data flow during training**: `train_ds`/`val_ds` are `PromoterDataset` instances (`src/dataset.py`) wrapping the CSVs, tokenizing each sequence to fixed-length `input_ids`/`attention_mask`/`token_type_ids` plus the integer `labels`. Training touches only `train.csv`; validation (after every epoch, no gradient updates) touches only `val.csv`; the test set is never opened during this phase at all.

**Path robustness**: all three scripts (`train.py`, `evaluate.py`, `predict.py`) resolve `data/`, `checkpoints/`, `results/` relative to the *repository root* (computed from the script's own file location via `src/paths.py`), not relative to whatever directory the shell happens to be in -- this was a real, repeatedly-hit bug class (Colab notebook cells `%cd`-ing into `src/` before invoking, causing `FileNotFoundError` against a `data/` that didn't exist relative to the current directory) closed off by this design rather than by remembering to `cd` correctly every time.

---

## 9. Evaluation (`src/evaluate.py`)

Run as: `python src/evaluate.py --seq-len 1000 --output results/test_results.txt`

Loads the saved checkpoint, runs inference (no gradients, `model.eval()`) over the **test set only**, and reports `sklearn`'s `classification_report` (precision/recall/F1 per class + macro/weighted averages) and confusion matrix. **The test set is opened exactly once per experiment, purely to report a final number** -- never used to pick hyperparameters, never used to pick which checkpoint to keep (that's what validation is for). This ordering is what makes the reported test metrics trustworthy rather than optimistically biased.

---

## 10. Results: baseline and every experiment tried

All numbers on the held-out test set (n=599), computed from the confusion matrix for internal consistency (`src/plot_results.py`):

| Configuration | Accuracy | Macro F1 | F1 (Low) | F1 (High) | Verdict |
|---|---|---|---|---|---|
| **Baseline** (full fine-tune, lr=3e-5, 5 epochs, early-stopped at epoch 3, best=epoch 1) | 0.676 | 0.668 | 0.617 | 0.720 | Reference |
| **Baseline + tuned decision threshold (0.66)** | **0.690** | **0.687** | 0.661 | 0.714 | **Best -- shipped in `predict.py`** |
| lr=1e-5 (full fine-tune) | 0.676 | 0.669 | 0.620 | 0.718 | No real difference (noise-level) |
| Frozen-base (only classifier + pooler trained) | 0.641 | 0.636 | 0.591 | 0.681 | Worse -- full fine-tuning matters |
| Attention dropout=0.1 | 0.633 | 0.619 | 0.547 | 0.691 | Worse -- over-regularized for this training budget |

See `results/plots/confusion_matrices.png`, `results/plots/config_comparison.png`, and `results/plots/epoch_progression.png` for the visual versions, and `RESULTS_EXPLAINED.md` for what each metric means in this project's specific context.

**The experimental narrative, honestly stated:** four reasoned hypotheses were tested against the baseline. One (decision-threshold tuning) produced a real, generalizing improvement, requiring zero retraining. Two (learning rate, attention dropout) produced negative or null results -- legitimate findings in their own right, not failures of the process. One (frozen vs. fine-tuned) directly answered a stated research question: full fine-tuning of DNABERT-2's weights meaningfully outperforms using it as a frozen feature extractor.

---

## 11. Validation methodology (train/val/test discipline)

| Split | File | Size | Role |
|---|---|---|---|
| Train | `data/train_1000bp.csv` | 4,796 | Weight updates via backprop |
| Validation | `data/val_1000bp.csv` | 600 (500 subsampled by default in `train.py`) | Checkpoint selection + early stopping after every epoch -- never trains on this |
| Test | `data/test_1000bp.csv` | 599 | Opened exactly once, after training is fully finished, purely to report a final number |

Cross-split `gene_id` overlap is verified at 0 (see section 4). The decision threshold used in the final `predict.py` (0.66) was selected by sweeping macro F1 **on the validation set only** (`src/tune_threshold.py`), then applied blind to the test set -- and it improved test macro F1 too (0.668 -> 0.687), which is the evidence it generalizes rather than being an artifact of tuning on the wrong data.

---

## 12. Repository structure

```
GenePromoter/
  data/
    raw_data.csv              # joined GTEx + Ensembl data, leak-free (5996 genes)
    train_1000bp.csv          # 4796 genes
    val_1000bp.csv            # 600 genes
    test_1000bp.csv           # 599 genes
  src/
    fetch_promoter_data.py    # Phase 1: build raw_data.csv from GTEx + Ensembl
    preprocess.py             # Phase 2: truncate/label/split
    dataset.py                # PromoterDataset: CSV -> tokenized tensors
    patch_dnabert2.py         # the 4-issue compatibility patch, called by every model-touching script
    check_tokenization.py     # Phase 3: token-length sanity check
    train.py                  # Phase 4: fine-tuning loop (+ --freeze-base, --lr, --attn-dropout flags)
    evaluate.py                # Phase 5: test-set metrics
    predict.py                 # Phase 6: standalone inference (the CC handoff artifact)
    tune_threshold.py          # post-hoc decision threshold tuning (no retraining)
    paths.py                   # repo-root-relative path resolution
    eda.py                     # exploratory data analysis (plots + written inference)
    plot_results.py            # final comparison/confusion-matrix plots
  eda/
    eda_summary.png / .md      # EDA plots + written analysis
  results/
    test_results*.txt          # per-experiment test metrics (gitignored -- regenerable)
    threshold_tuning.txt       # threshold sweep + before/after comparison (gitignored)
    plots/                     # confusion matrices, config comparison, epoch progression (tracked)
  checkpoints/
    best_model/                # trained weights (gitignored -- too large for git; see DL_HANDOFF.md)
  notebooks/
    train_colab.ipynb          # the full Colab GPU workflow: clone -> install -> sanity check -> train -> eval -> experiments -> download
  dl_roadmap.md                 # the original phase-by-phase DL execution plan
  requirements.txt              # pinned versions for local/CPU use (NOT used as-is on Colab -- see notebook Cell 2's reasoning)
```

---

## 13. How to reproduce from zero

```bash
# 1. Environment (local, CPU-capable sanity checks only -- see DL_HANDOFF.md for why training needs a GPU)
python -m venv .venv
source .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Data (only if regenerating from scratch -- committed CSVs already exist)
python src/fetch_promoter_data.py        # -> data/raw_data.csv
python src/preprocess.py --seq-len 1000  # -> data/{train,val,test}_1000bp.csv

# 3. EDA
python src/eda.py --seq-len 1000

# 4. Training -- needs a GPU in practice; see notebooks/train_colab.ipynb for the full Colab workflow
python src/train.py --seq-len 1000 --max-train-samples -1 --epochs 5

# 5. Evaluation
python src/evaluate.py --seq-len 1000 --output results/test_results.txt

# 6. Threshold tuning (no retraining)
python src/tune_threshold.py --seq-len 1000

# 7. Inference
python src/predict.py ACGTACGTACGT...

# 8. Result plots
python src/plot_results.py
```
