# Model Choice: Defense Notes (Q&A)

This document captures the reasoning behind key design decisions in this project, in the form they're most likely to actually come up: as questions from a reviewer or faculty member. Each answer is self-contained -- meant to be read (or spoken from) directly in a viva/defense setting, not just skimmed as background.

---

## Q: Why DNABERT-2 specifically, and not some other model?

Three converging reasons, not just "it was the first one found":

1. **It's a genomic-domain pretrained model, not a general-purpose one.** DNABERT-2 was pretrained via masked-language-modeling directly on DNA sequences across multiple species, so it already carries a general statistical understanding of DNA structure -- recurring motifs, base-transition patterns -- before this project ever touches it. A general-purpose language model (or a vision/generic model) would have none of that.
2. **It's built for exactly this kind of downstream workflow.** It's packaged on HuggingFace with `trust_remote_code` support and is directly compatible with `AutoModelForSequenceClassification` -- attach a classification head, fine-tune on a custom labeled dataset. That's precisely what this project needed: take a pretrained model, adapt it to a new binary classification task with a relatively small labeled set (~4,800 training genes).
3. **It uses Byte-Pair Encoding (BPE) rather than fixed k-mers**, unlike the original DNABERT. This lets it form variable-length tokens the way a language model forms subwords, which in principle lets it represent motifs of varying natural length more efficiently than a rigid fixed-k-mer tokenizer would.

---

## Q: Why not train a model from scratch instead of fine-tuning a pretrained one?

Because the dataset is small -- ~4,800 training examples after the leak-free train/val/test split. A 117M-parameter Transformer trained from scratch on that little data would almost certainly overfit badly: it has far more capacity than the data can constrain, and would likely memorize the training set rather than learn generalizable sequence patterns. This is precisely the standard justification for transfer learning in low-data regimes -- start from weights that already encode general structure (learned from a much larger pretraining corpus than this project could ever assemble), then adapt only what's needed for the specific task. Training from scratch here would not be a more rigorous approach; it would very likely be a strictly worse one, given the data available.

---

## Q: Why a 1000bp promoter window, and not something longer?

Two separate reasons, one a project-scoping choice and one a hard architectural constraint:

**Scoping reason:** the project's own execution plan explicitly chose 1000bp as a pragmatic "manageable first run" starting point, with window-size comparison (500bp/2000bp) deliberately deferred as optional follow-up research rather than decided upfront by exhaustive search.

**Architectural reason:** DNABERT-2's own config (`config.json`) sets `max_position_embeddings: 512`. At roughly 220 BPE tokens per 1000bp of sequence, that puts the model's practical ceiling at approximately **2,000-2,300bp** -- going meaningfully beyond that pushes past what the model was pretrained and positionally designed for. 1000bp sits comfortably inside that budget with room to spare (verified via `check_tokenization.py`: mean 204 tokens, p95 217, well under the 256-token budget used); 2000bp (already-fetched, no re-fetch needed) approaches the model's real limit, which is exactly why it was scoped as the stretch-goal comparison rather than the default.

**Why not just increase it arbitrarily:** standard Transformer self-attention cost scales *quadratically* with token count. Doubling the sequence length roughly quadruples the attention compute cost per layer, not just doubles it -- so "just use a bigger window" is not compute-free even within the range the model can nominally support, and becomes outright infeasible well before reaching the kind of context genome-wide expression models like Enformer use (see below).

---

## Q: Given promoter sequence alone is known to be an incomplete predictor of expression (enhancers, chromatin state, etc. are elsewhere), why not use a model designed for that, like Enformer?

Because that would be a fundamentally different, much larger project, not a drop-in improvement within this project's scope:

1. **Enformer is not casually fine-tunable.** It was pretrained on TPU infrastructure to predict thousands of genome-wide epigenomic/expression tracks (CAGE, DNase, ChIP-seq) across ~200,000bp windows. Its public tooling is built around using its existing pretrained predictions directly, not around "attach a new head and fine-tune on my own small custom binary-labeled dataset" -- adapting it for this task would require substantial custom architecture work, unlike DNABERT-2 which already had a ready `AutoModelForSequenceClassification` path.
2. **The context-length gap isn't a settings change.** Enformer's ~200kb effective context is achieved through a fundamentally different architecture (dilated convolutions compressing long-range sequence context before attention), not by scaling up a standard Transformer's attention window -- because quadratic self-attention over 200kb of raw sequence (tens of thousands of tokens) is computationally impractical on the free-tier GPU resources this project used.
3. **Data acquisition cost scales with it.** A 200kb-per-gene window is 200x more sequence per gene than the current 1000bp window. The Ensembl fetch was already a real, documented bottleneck (slow, rate-limited) at 2000bp; scaling that up 200x would be a substantially larger data-engineering undertaking on its own.
4. **It doesn't fit the stated project scope.** The project was explicitly scoped (from the original project overview) to be feasible within a limited timeline using fine-tuning rather than large-scale pretraining or training from scratch, specifically because fine-tuning "requires significantly less computational effort... making the project suitable for a final-year project." Enformer-scale work is a different order of magnitude of effort, not a parameter to swap in.

**The honest, defensible position:** promoter-sequence-only prediction has a real, known ceiling because true regulatory control also involves distal elements outside any promoter-only window -- this is stated explicitly as a limitation in `RESULTS_EXPLAINED.md` and `PROJECT_DOCUMENTATION.md`, not something discovered only when asked. A model like Enformer would likely do better *because it sees more of the regulatory landscape*, not because it's a "better" architecture in the abstract -- and that's exactly the right thing to name as a **future work direction**, showing the limitation was understood and reasoned about, rather than something to have naively built from the start given the actual time and compute available.

---

## Q: So is 68-69% accuracy actually a good result, or should it have been higher?

It's a legitimate, defensible result for what the task structurally allows, not a shortfall to apologize for:

- **Chance baseline is 50%** (balanced binary classes) -- the model clears it by a real margin on both accuracy and macro F1.
- **A trivial sequence statistic (GC content) explains only ~2.2% of expression variance** (measured directly in this project's own EDA, `eda/eda_summary.md`) -- so a model relying on nothing more than composition would land only marginally above chance. DNABERT-2 reaching 68-69% is evidence it's using real sequence structure beyond simple composition.
- **The task has a known, structural ceiling below 100%** regardless of model quality, because promoter sequence alone omits distal enhancers, chromatin accessibility, and tissue-specific transcription factor levels -- all real contributors to expression that simply aren't present in the input by construction (see the Enformer discussion above). A 65-75% range for a promoter-sequence-only expression classifier is a normal, respectable outcome in the genomics-DL literature, not evidence of an under-performing pipeline.
- **Four separate improvement hypotheses were tested honestly** (lower LR, frozen-base, attention dropout, decision-threshold tuning) rather than just accepting the first result -- one produced a real, validated improvement (threshold tuning: macro F1 0.668 -> 0.687), the others produced legitimate negative results that are documented, not hidden. That experimental discipline -- not the raw accuracy number -- is the actual evidence of a well-executed project.
