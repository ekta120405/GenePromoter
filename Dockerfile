# GenePromoter inference service: DNABERT-2 behind a FastAPI REST API.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. Dependencies first, so Docker caches this layer and code edits don't
#    trigger a multi-minute torch reinstall.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 2. Model checkpoint (~470MB) and the DL side's inference code.
COPY checkpoints/ ./checkpoints/
COPY src/predict.py src/patch_dnabert2.py src/paths.py ./src/

# 3. Bake the HuggingFace cache into the image at BUILD time: downloads the
#    DNABERT-2 base repo + tokenizer, applies patch_dnabert2's fixes, and
#    loads the model once. The running container then needs no internet,
#    so the image is self-contained and every Pod starts identically.
#    The base repo's own weights (~470MB) are then deleted: the fine-tuned
#    weights come from checkpoints/best_model, only the tokenizer and code
#    files are used from the base repo.
RUN python -c "import sys; sys.path.insert(0, 'src'); from predict import load; load(); print('model cache baked')" \
 && find /root/.cache/huggingface/hub -path '*DNABERT-2-117M/snapshots/*' \( -name '*.bin' -o -name '*.safetensors' \) \
      -exec sh -c 'rm -f "$(readlink -f "$1")" "$1"' _ {} \; \
 && du -sh /root/.cache/huggingface
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# 4. API code last: editing app.py or the web page only rebuilds this layer.
COPY app.py .
COPY static/ ./static/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
