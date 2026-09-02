"""
DNABERT-2's remote code (zhihan1996/DNABERT-2-117M) predates modern
`transformers` and breaks on any current install in two ways:

1. `bert_layers.py` imports `.flash_attn_triton`, which unconditionally
   imports `triton`. Transformers' `check_imports` statically scans this
   import chain and refuses to load the model if `triton` isn't installed
   -- even though `bert_layers.py` itself already falls back to standard
   attention when the import fails. There's no CPU/non-flash-attention
   wheel for `triton` on Windows, so we remove the import outright and
   force the standard-attention fallback.
2. None of `BertModel` / `BertForMaskedLM` / `BertForSequenceClassification`
   in `bert_layers.py` declare `config_class`, so they inherit it from the
   real `transformers.models.bert.modeling_bert.BertPreTrainedModel`
   (built-in `BertConfig`). `AutoModel.from_pretrained` then finds a
   mismatch against the repo's own custom `BertConfig` and raises
   `ValueError` during auto-registration. We patch in the correct
   `config_class` on each.

This module downloads the model repo's files (without instantiating the
model, so the triton ImportError never fires) and patches the cached
snapshot in place. Idempotent -- safe to call every run.
"""
import re
from huggingface_hub import snapshot_download

MODEL_ID = "zhihan1996/DNABERT-2-117M"

TRITON_BLOCK = """try:
    from .flash_attn_triton import flash_attn_qkvpacked_func
except ImportError as e:
    flash_attn_qkvpacked_func = None"""

TRITON_REPLACEMENT = """# triton-based flash attention is unavailable on CPU; force the standard-attention path.
flash_attn_qkvpacked_func = None"""

CONFIG_IMPORT_MARKER = "from .configuration_bert import BertConfig"


def ensure_patched():
    snapshot_dir = snapshot_download(MODEL_ID)
    path = f"{snapshot_dir}/bert_layers.py"
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    changed = False

    if TRITON_BLOCK in src:
        src = src.replace(TRITON_BLOCK, TRITON_REPLACEMENT)
        changed = True

    if CONFIG_IMPORT_MARKER not in src:
        src = src.replace(
            "from transformers.modeling_utils import PreTrainedModel",
            "from transformers.modeling_utils import PreTrainedModel\n\n"
            + CONFIG_IMPORT_MARKER,
            1,
        )
        changed = True

    for cls in ("BertModel(BertPreTrainedModel)", "BertForMaskedLM(BertPreTrainedModel)",
                "BertForSequenceClassification(BertPreTrainedModel)"):
        if f"class {cls}:\n\n    config_class = BertConfig" not in src:
            pattern = re.compile(rf"(class {re.escape(cls)}:.*?\n)(\s*)(def __init__)", re.DOTALL)
            def _inject(m):
                return f"{m.group(1)}{m.group(2)}config_class = BertConfig\n\n{m.group(2)}{m.group(3)}"
            new_src, n = pattern.subn(_inject, src, count=1)
            if n:
                src = new_src
                changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)

    return snapshot_dir


if __name__ == "__main__":
    print("Patched snapshot at:", ensure_patched())
