import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PromoterDataset
from patch_dnabert2 import ensure_patched
from paths import repo_path

MODEL_ID = "zhihan1996/DNABERT-2-117M"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1000, help="promoter window length; must match preprocess.py output")
    ap.add_argument("--max-length", type=int, default=256, help="BPE token budget per sequence")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-train-samples", type=int, default=2000,
                     help="subsample the training set for a fast CPU run; pass -1 for the full set")
    ap.add_argument("--max-val-samples", type=int, default=500)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--output-dir", default="checkpoints/best_model")
    ap.add_argument("--freeze-base", action="store_true",
                     help="freeze all but the classification head (stretch-goal experiment)")
    ap.add_argument("--attn-dropout", type=float, default=None,
                     help="override attention_probs_dropout_prob (DNABERT-2 pretrains with 0.0, "
                          "originally to stay flash-attention-eligible -- moot here since triton/"
                          "flash-attention is unavailable and patched out regardless); e.g. 0.1")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_patched()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.num_labels = 2
    if args.attn_dropout is not None:
        config.attention_probs_dropout_prob = args.attn_dropout
        print(f"attention_probs_dropout_prob overridden to {args.attn_dropout}")

    # low_cpu_mem_usage=False: DNABERT-2's custom BertEncoder.__init__ builds an
    # ALiBi tensor eagerly (rebuild_alibi_tensor), which breaks under newer
    # transformers' default meta-device fast-init path (tensors it allocates
    # land on "meta" while others land on "cpu" -> device mismatch).
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, config=config, trust_remote_code=True, low_cpu_mem_usage=False
    )
    model.to(device)

    if args.freeze_base:
        # bert.pooler.dense is also freshly initialized (never in the
        # pretrained checkpoint -- it always shows up as MISSING in the
        # load report), and the classifier reads from its output
        # (pooled_output = outputs[1] -> classifier(pooled_output)). If
        # left frozen at random init, the classifier would be learning on
        # top of an untrained random projection, not a meaningful
        # "frozen features" baseline -- so both heads stay trainable here.
        for name, p in model.named_parameters():
            if not (name.startswith("classifier") or name.startswith("bert.pooler")):
                p.requires_grad = False

    suffix = f"_{args.seq_len}bp"
    train_ds = PromoterDataset(repo_path("data", f"train{suffix}.csv"), tokenizer, max_length=args.max_length)
    val_ds = PromoterDataset(repo_path("data", f"val{suffix}.csv"), tokenizer, max_length=args.max_length)

    if args.max_train_samples >= 0 and args.max_train_samples < len(train_ds):
        train_ds.df = train_ds.df.sample(n=args.max_train_samples, random_state=42).reset_index(drop=True)
    if args.max_val_samples >= 0 and args.max_val_samples < len(val_ds):
        val_ds.df = val_ds.df.sample(n=args.max_val_samples, random_state=42).reset_index(drop=True)

    print(f"train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=50, num_training_steps=args.epochs * len(train_loader)
    )

    output_dir = repo_path(args.output_dir)
    best_val_f1 = 0.0
    patience_counter = 0
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds += outputs.logits.argmax(-1).cpu().tolist()
                labels += batch["labels"].cpu().tolist()
        val_f1 = f1_score(labels, preds)
        val_acc = accuracy_score(labels, preds)
        print(f"Epoch {epoch}: train_loss={total_loss / len(train_loader):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  saved new best checkpoint to {output_dir}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered")
                break

    print(f"Best val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
