"""
Final result visualizations for the paper/documentation: confusion matrices
for every experiment, a config comparison bar chart, and the baseline
run's per-epoch training curve. Numbers are hardcoded from the actual
training/evaluation runs (see results/*.txt and the project documentation)
rather than re-parsed from text reports, to keep this simple and exact.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "plots")
os.makedirs(OUTDIR, exist_ok=True)

# confusion matrix rows=true [Low, High], cols=pred [Low, High]
CONFIGS = {
    "Baseline\n(lr=3e-5)": np.array([[156, 124], [70, 249]]),
    "Baseline + tuned\nthreshold (0.66)": np.array([[181, 99], [87, 232]]),
    "lr=1e-5": np.array([[158, 122], [72, 247]]),
    "Frozen-base": np.array([[155, 125], [90, 229]]),
    "Attn-dropout=0.1": np.array([[133, 147], [73, 246]]),
}


def metrics_from_cm(cm):
    a, b, c, d = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]  # a=TN(Low/Low) b=FP c=FN d=TP(High/High)
    accuracy = (a + d) / cm.sum()
    prec_low, rec_low = a / (a + c), a / (a + b)
    prec_high, rec_high = d / (b + d), d / (c + d)
    f1_low = 2 * prec_low * rec_low / (prec_low + rec_low)
    f1_high = 2 * prec_high * rec_high / (prec_high + rec_high)
    macro_f1 = (f1_low + f1_high) / 2
    return {"accuracy": accuracy, "macro_f1": macro_f1, "f1_low": f1_low, "f1_high": f1_high}


# ---- Figure 1: confusion matrices, one panel per config ----
fig, axes = plt.subplots(1, len(CONFIGS), figsize=(4 * len(CONFIGS), 4.2))
for ax, (name, cm) in zip(axes, CONFIGS.items()):
    m = metrics_from_cm(cm)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Low", "High"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Low", "High"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name}\nacc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}", fontsize=10)
fig.suptitle("Confusion matrices -- all experiments (test set, n=599)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/confusion_matrices.png", dpi=150)
print(f"Saved {OUTDIR}/confusion_matrices.png")

# ---- Figure 2: config comparison bar chart ----
names = list(CONFIGS.keys())
accs = [metrics_from_cm(cm)["accuracy"] for cm in CONFIGS.values()]
f1s = [metrics_from_cm(cm)["macro_f1"] for cm in CONFIGS.values()]

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(names))
w = 0.35
bars1 = ax.bar(x - w / 2, accs, w, label="Accuracy", color="#4C72B0")
bars2 = ax.bar(x + w / 2, f1s, w, label="Macro F1", color="#55A868")
ax.set_xticks(x); ax.set_xticklabels([n.replace("\n", " ") for n in names], rotation=20, ha="right")
ax.set_ylim(0, 0.8)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (0.5)")
ax.set_title("Test-set accuracy and macro F1 across all tried configurations")
ax.legend()
for bars in (bars1, bars2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}",
                 ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/config_comparison.png", dpi=150)
print(f"Saved {OUTDIR}/config_comparison.png")

# ---- Figure 3: baseline per-epoch training curve ----
epochs = [0, 1, 2, 3]
train_loss = [0.6577, 0.6229, 0.5495, 0.4344]
val_acc = [0.6420, 0.6520, 0.6440, 0.6360]
val_f1 = [0.6786, 0.6947, 0.6229, 0.6389]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(epochs, train_loss, marker="o", color="#C44E52", label="train_loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss", color="#C44E52")
ax1.tick_params(axis="y", labelcolor="#C44E52")
ax1.set_xticks(epochs)

ax2 = ax1.twinx()
ax2.plot(epochs, val_acc, marker="s", color="#4C72B0", label="val_acc")
ax2.plot(epochs, val_f1, marker="^", color="#55A868", label="val_f1")
ax2.set_ylabel("Validation score")
ax2.axvline(1, color="gray", linestyle=":", linewidth=1)
ax2.annotate("best checkpoint\n(early stopping)", xy=(1, val_f1[1]), xytext=(1.9, 0.60),
             arrowprops=dict(arrowstyle="->"), fontsize=9)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
ax1.set_title("Baseline training run: train loss keeps falling while val plateaus at epoch 1")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/epoch_progression.png", dpi=150)
print(f"Saved {OUTDIR}/epoch_progression.png")

print("\nAll metrics (recomputed from confusion matrices, for consistency):")
for name, cm in CONFIGS.items():
    m = metrics_from_cm(cm)
    print(f"  {name.replace(chr(10), ' ')}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
          f"f1_low={m['f1_low']:.4f} f1_high={m['f1_high']:.4f}")
