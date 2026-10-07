"""
GenePromoter Inference API.

Wraps src/predict.py (from the DL side) in a REST API. The model is loaded
ONCE at startup, not per request, so every /predict call reuses it.

Endpoints
  GET  /          simple web page to paste a sequence and get a prediction
  GET  /health    liveness/readiness check used by the Kubernetes probes
  GET  /info      model + deployment metadata (useful in the demo)
  POST /predict   {"sequence": "ACGT..."} -> {"prediction", "confidence", "label", "probability_high", ...}
  GET  /docs      auto-generated Swagger UI (FastAPI built-in)
"""
import os
import re
import socket
import sys
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from predict import HIGH_THRESHOLD, load, predict as run_predict  # noqa: E402

MIN_LEN = 50      # anything shorter is not a meaningful promoter window
MAX_LEN = 5000    # model only sees ~1000bp anyway (256 tokens); cap abuse
DNA_RE = re.compile(r"^[ACGT]+$")

state = {"started_at": None, "load_seconds": None, "requests_served": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    tokenizer, model = load()
    state["tokenizer"], state["model"] = tokenizer, model
    state["load_seconds"] = round(time.time() - t0, 1)
    state["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[startup] model loaded in {state['load_seconds']}s (torch threads: {torch.get_num_threads()})", flush=True)
    yield
    state.clear()


app = FastAPI(
    title="GenePromoter Inference API",
    description="DNABERT-2 fine-tuned to predict HIGH/LOW gene expression (liver) from a promoter DNA sequence.",
    version="1.0.0",
    lifespan=lifespan,
)


class SequenceRequest(BaseModel):
    sequence: str = Field(..., description="Promoter DNA sequence (A/C/G/T), ideally ~1000bp TSS-centred")

    @field_validator("sequence")
    @classmethod
    def clean_and_check(cls, v: str) -> str:
        v = re.sub(r"\s+", "", v).upper()   # tolerate pasted line breaks / lowercase
        if not v:
            raise ValueError("sequence is empty")
        if not DNA_RE.match(v):
            raise ValueError("sequence must contain only A, C, G, T")
        if not MIN_LEN <= len(v) <= MAX_LEN:
            raise ValueError(f"sequence length must be between {MIN_LEN} and {MAX_LEN} bp (got {len(v)})")
        return v


class PredictionResponse(BaseModel):
    prediction: str           # DL contract: "HIGH EXPRESSION" / "LOW EXPRESSION"
    confidence: float         # DL contract: 0-100, probability of the predicted label
    label: str                # "HIGH" / "LOW"
    probability_high: float   # P(HIGH), 0-1; HIGH when >= threshold
    threshold: float
    sequence_length: int
    inference_ms: float
    served_by: str


app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


@app.get("/health")
def health():
    loaded = "model" in state
    if not loaded:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return {"status": "ok", "model_loaded": True}


@app.get("/info")
def info():
    return {
        "model": "DNABERT-2-117M fine-tuned (liver expression)",
        "decision_threshold_high": HIGH_THRESHOLD,
        "input_window_bp": 1000,
        "pod_name": socket.gethostname(),   # in Kubernetes this is the Pod name
        "started_at": state.get("started_at"),
        "model_load_seconds": state.get("load_seconds"),
        "requests_served": state.get("requests_served", 0),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(req: SequenceRequest):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    t0 = time.time()
    result = run_predict(req.sequence, tokenizer=state["tokenizer"], model=state["model"])
    state["requests_served"] += 1
    label = result["prediction"].split()[0]
    p = result["confidence"] / 100
    return {
        **result,
        "label": label,
        "probability_high": round(p if label == "HIGH" else 1 - p, 3),
        "threshold": HIGH_THRESHOLD,
        "sequence_length": len(req.sequence),
        "inference_ms": round((time.time() - t0) * 1000, 1),
        "served_by": socket.gethostname(),
    }
