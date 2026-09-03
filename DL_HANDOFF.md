# DL -> CC Handoff

This is the handoff artifact: what the Deep Learning side is delivering, exactly how to call it, what it needs to run, and what its known limits are. The Cloud Computing side should be able to build the Docker/Kubernetes deployment from this document alone, without needing to understand the training process (that's `PROJECT_DOCUMENTATION.md`, for reference/the paper, not required reading to do the deployment).

---

## What's being handed off

1. **A trained model checkpoint**: `checkpoints/best_model/` (~470MB). Contains the fine-tuned DNABERT-2 weights (`model.safetensors`), its config, and the model's custom remote code (`bert_layers.py`, `configuration_bert.py`, `bert_padding.py`) bundled alongside so the checkpoint is self-contained.
   - **This is not committed to git** (too large, and checkpoints are generated artifacts, not source). It needs to be transferred separately -- see "Getting the checkpoint" below.
2. **A standalone inference script**: `src/predict.py`. This is the only script the CC side needs to actually run inference -- it does not depend on any of the training/data-pipeline scripts.
3. **A pinned dependency list**: `requirements.txt`.

---

## Getting the checkpoint

The checkpoint isn't in git. It currently exists as a local file on the DL side's machine, produced by downloading `checkpoints/best_model/` from the Colab training session (zipped, downloaded via browser, extracted locally). For the CC side to get it:
- Simplest: DL side shares the `checkpoints/best_model/` folder directly (zip it, send via whatever transfer method the team uses -- it's ~470MB, too large for git, fine for direct transfer).
- Alternative: re-run `notebooks/train_colab.ipynb` end to end on Colab and download the checkpoint from there directly.

Whichever way it arrives, it needs to end up at `checkpoints/best_model/` relative to wherever `predict.py` is being run from (see "Directory layout" below).

---

## Exact input/output contract

**`predict.py` exposes one function:**

```python
def predict(seq: str, tokenizer=None, model=None, max_length=256) -> dict
```

**Input**: `seq` -- a single DNA sequence, as a plain Python string. Expected to be uppercase `A`/`C`/`G`/`T` characters (the tokenizer will tokenize whatever string it's given; sequences containing other characters were never seen during training and will produce unpredictable/unreliable behavior, not necessarily an error). No fixed length is enforced by the function signature, but the tokenizer truncates to `max_length=256` tokens -- roughly the first ~1000bp of content is what the model actually "sees" for a longer input (matches training: the model was trained on exactly-1000bp windows). Passing dramatically shorter or longer sequences than that is untested territory, not something the model was validated against.

**Output**: a dict with exactly two keys:
```python
{"prediction": "HIGH EXPRESSION", "confidence": 68.7}
```
- `prediction`: the literal string `"HIGH EXPRESSION"` or `"LOW EXPRESSION"`.
- `confidence`: a float, 0-100, the model's probability (as a percentage) for whichever label was actually predicted -- not simply "the higher of the two class probabilities" (those diverge once a non-0.5 decision threshold is in play, which it is here -- see below).

**Command-line usage** (for manual testing): `python src/predict.py <DNA_SEQUENCE>`, prints the same dict.

**The decision threshold is 0.66, not the naive 0.5.** This is baked into `predict.py` as the `HIGH_THRESHOLD` constant, derived empirically (`src/tune_threshold.py`, swept on the validation set, improved test macro F1 from 0.668 to 0.687 -- see `PROJECT_DOCUMENTATION.md` section 10). **If the model is ever retrained, this constant needs to be re-derived** by re-running `tune_threshold.py` against the new checkpoint -- it's specific to this exact trained model, not a universal constant.

---

## Directory layout `predict.py` expects

```
<repo root>/
  checkpoints/
    best_model/
      model.safetensors
      config.json
      configuration_bert.py
      bert_layers.py
      bert_padding.py
      tokenizer.json
      tokenizer_config.json
  src/
    predict.py
    patch_dnabert2.py     # predict.py imports this -- must sit alongside it
    paths.py              # predict.py imports this too
```

`predict.py` resolves `checkpoints/best_model` relative to the **repository root** (computed from its own file location, not the current working directory -- see `src/paths.py`), so it can be invoked from anywhere as long as this directory structure around it is intact. **`src/patch_dnabert2.py` and `src/paths.py` must be copied alongside `predict.py`** -- it's not a fully standalone single file, it has these two same-directory dependencies.

---

## What happens on first call, mechanically

1. `ensure_patched()` (from `patch_dnabert2.py`) downloads the base DNABERT-2 model repo's files from HuggingFace Hub (if not already cached) and patches known compatibility issues into the cached copy -- see `PROJECT_DOCUMENTATION.md` section 7 for the full list of what's being patched and why. **This means the very first call needs internet access** (to reach `huggingface.co`) even though the actual fine-tuned weights are loaded from the local checkpoint, not downloaded. Subsequent calls reuse the HuggingFace cache (`~/.cache/huggingface/`) and don't re-download.
2. The tokenizer is loaded from the **base model repo** (`zhihan1996/DNABERT-2-117M` on HuggingFace Hub), not from the checkpoint directory -- this is deliberate, not a bug: fine-tuning never changes the tokenizer, and loading it from the checkpoint risks a `tokenizer_class` incompatibility if the checkpoint was saved by a different `transformers` version than what's installed at inference time (this was hit and fixed during development -- see `PROJECT_DOCUMENTATION.md` section 7 discussion, or `predict.py`'s own comments).
3. The fine-tuned model weights are loaded from the local `checkpoints/best_model/` directory.
4. Inference runs on GPU if `torch.cuda.is_available()`, otherwise CPU automatically -- no code change needed either way, but see performance notes below.

---

## Performance characteristics (what to expect for the Docker/K8s deployment)

- **Model size on disk**: ~470MB (mostly `model.safetensors`).
- **CPU inference works and was verified locally** (single-sequence predictions complete in a few seconds on a standard CPU) -- a GPU is *not* required for serving predictions, only for training. This matters for the CC side's resource sizing: a CPU-only container is a legitimate, tested option for the deployed inference service, even though training itself needed a GPU (Colab T4).
- **No batching implemented** -- `predict()` handles one sequence per call. If the CC side needs throughput beyond single-request-at-a-time, that's an enhancement to build on top of `predict.py`'s `predict()` function (which is already factored to accept a pre-loaded `tokenizer`/`model` so repeated calls don't reload the model each time -- see its signature).
- **First-call latency will be higher** than subsequent calls, due to the HuggingFace cache population / patch step described above. Worth loading the model once at container startup (e.g. in FastAPI's startup event) rather than per-request.

---

## Dependencies (`requirements.txt`)

```
torch==2.14.0
transformers==4.40.0
einops==0.8.2
accelerate==1.14.0
scikit-learn==1.9.0
pandas==3.0.5
numpy==2.5.2
datasets==5.0.1
requests==2.34.2
matplotlib==3.11.1
```

**For the CC Docker image, only `torch`, `transformers`, `einops` are actually needed by `predict.py` at runtime** (`scikit-learn`/`pandas`/`datasets`/`matplotlib` are used by the training/EDA/plotting scripts, not inference). Trimming the Docker image's dependency list to just what `predict.py` imports is a reasonable, safe optimization -- check `predict.py`'s own imports to confirm the minimal set before trimming.

**Torch version note**: this pin (`2.14.0`) was chosen for a CPU-only Windows dev machine. For a Docker image, use whatever official PyTorch CPU (or CUDA, if GPU-serving is desired) wheel matches the target Python version -- don't assume this exact pin is optimal for a Linux container; verify a matching wheel exists.

---

## Known limitations (be aware, not necessarily blockers)

- **Tissue-specific model**: trained on Liver expression labels only. It answers "is this gene likely highly/lowly expressed in liver," not expression in general or in any other tissue.
- **Fixed input window**: trained on ~1000bp promoter windows. Sequences very different in nature from a real TSS-centered promoter window (e.g. random genomic DNA, coding sequence, a different species) are out-of-distribution -- the model will still return a prediction (it can't detect "this doesn't look like a promoter"), but that prediction has no validated meaning for such input.
- **Accuracy ceiling**: ~69% test accuracy / macro F1. See `RESULTS_EXPLAINED.md` for why this is a legitimate, expected range for this task rather than a bug to chase further before deployment.
- **No confidence calibration guarantee beyond what's measured**: the reported `confidence` is the model's raw softmax probability for the predicted class, not independently calibrated (e.g. via temperature scaling) -- treat it as a relative signal (higher = more confident) rather than a literal well-calibrated probability.
