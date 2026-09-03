# Final Results Explained

The final, best configuration: DNABERT-2 fine-tuned on the full training set (baseline weights), with the decision threshold tuned to 0.66 (see `PROJECT_DOCUMENTATION.md` sections 10-11 for how this was arrived at). This document explains what each reported metric actually means, both in general and specifically for this project, and what the final numbers say about the model.

---

## The final numbers

Test set (n=599 genes, never touched during training or checkpoint selection):

|  | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| **Low** | 0.68 | 0.65 | 0.66 | 280 |
| **High** | 0.70 | 0.73 | 0.71 | 319 |

**Accuracy: 0.69** | **Macro F1: 0.69** | **Weighted F1: 0.69**

Confusion matrix (rows = true label, columns = predicted label):
```
              Pred Low   Pred High
True Low         181         99
True High         87        232
```

---

## What each metric means, and what it means *here*

### Accuracy
**General:** the fraction of all predictions that were correct. `(correct predictions) / (total predictions)`.
**Here:** 0.69 means the model correctly called High-vs-Low for 69% of the 599 held-out genes. **The baseline to beat is 0.50** (a coin flip), not 0 -- this is a binary task with roughly balanced classes, so a model that always guesses one class would still score close to 50%. Accuracy alone can be misleading when classes are imbalanced (it wasn't here -- test set is 280 Low / 319 High, close to even), which is why the other metrics below matter too.

### Precision
**General:** of everything the model *predicted* as a given class, what fraction was actually that class. `TP / (TP + FP)`. Answers: "when the model says High, how often is it right?"
**Here:** High precision = 0.70 -- when the model predicts a gene is highly expressed, it's correct 70% of the time (232 of 331 High predictions were genuinely High). Low precision = 0.68 -- similar reliability when it predicts Low.

### Recall
**General:** of everything that *actually is* a given class, what fraction did the model correctly find. `TP / (TP + FN)`. Answers: "of all the truly High genes, how many did the model catch?"
**Here:** High recall = 0.73 -- the model correctly identifies 73% of genuinely highly-expressed genes (232 of 319). Low recall = 0.65 -- it catches 65% of genuinely lowly-expressed genes (181 of 280). **This gap (73% vs 65%) is the model's remaining bias**: it's still somewhat more inclined to call a gene "High" than "Low" when uncertain, even after threshold tuning narrowed this gap considerably from the untuned baseline (which was 78%/56% -- a much larger imbalance).

### F1-score
**General:** the harmonic mean of precision and recall for one class -- a single number that penalizes a model for being lopsided (e.g. very high recall but terrible precision, or vice versa) rather than rewarding either extreme alone.
**Here:** Low F1 = 0.66, High F1 = 0.71. The gap between them is smaller than the accuracy number alone would suggest, meaning the model isn't dramatically better at one class than the other -- there's a real but moderate imbalance, not a broken model that only learned one class.

### Macro average vs weighted average
**General:** *macro* average treats both classes equally regardless of how many examples of each exist (`(F1_Low + F1_High) / 2`). *Weighted* average weights each class's contribution by how many examples it has (`support`).
**Here:** they're both 0.69, essentially identical, because the test set is nearly balanced (280 vs 319) -- so class imbalance isn't distorting either number. **Macro F1 is the more meaningful single number to report** for this project specifically, since it doesn't let the slightly larger High class dominate the score.

### Confusion matrix
**General:** the full breakdown of every true-label/predicted-label combination, the rawest and most honest view of what the model gets right and wrong.
**Here:**
- **181 true negatives** (true Low, correctly predicted Low)
- **232 true positives** (true High, correctly predicted High)
- **99 false positives** (true Low, incorrectly predicted High) -- genes the model overestimates
- **87 false negatives** (true High, incorrectly predicted Low) -- genes the model underestimates

The fact that false positives (99) and false negatives (87) are now much closer to each other than in the untuned baseline (124 vs 70) is the concrete evidence that threshold tuning fixed a real, measurable bias, not just moved a number around cosmetically.

---

## What "getting it wrong" means biologically here

There's no safety-critical cost asymmetry in this specific task the way there might be in, say, medical diagnosis (where a false negative is far worse than a false positive). A misclassified gene here simply means: the promoter sequence alone didn't contain enough signal for the model to place that gene on the correct side of the (statistically defined) High/Low median split. Given that even the strongest simple sequence statistic found in EDA (GC content) explained only ~2.2% of expression variance, and real expression is also shaped by distal enhancers, chromatin state, and trans-acting factors entirely outside the ~1000bp window this model sees, some irreducible error rate is expected by construction -- not every misclassification represents a model failure, some represent genuinely insufficient information in the input itself.

---

## Putting the number in context

- **Chance baseline: 50%.** The model clears it by a meaningful margin (69% vs 50%), on both accuracy and macro F1.
- **GC-content correlation from EDA: r^2 ≈ 2.2% of variance.** A model relying on nothing more than sequence composition would be expected to land only modestly above chance. DNABERT-2 clearing 69% is evidence it's picking up more than simple composition statistics -- plausibly position-specific motifs and their arrangement relative to the TSS, which is exactly the kind of structure a Transformer's self-attention is suited to find and a single aggregate scalar like GC content cannot represent.
- **This is a genuinely hard task by construction**, not a toy problem the model should be expected to ace: promoter-sequence-only prediction of expression tier is missing a large share of the real regulatory picture (distal enhancers, chromatin accessibility, cell-type heterogeneity within "liver," tissue-specific transcription factor levels) that isn't present anywhere in the input. A 65-75% accuracy range for this class of task is a normal, respectable, publishable outcome in the genomics-DL literature -- not a sign of a broken pipeline.
