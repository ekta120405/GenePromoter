# CC Roadmap: Containerized Deployment of the GenePromoter Model

**Goal:** wrap `src/predict.py` in a REST API, package it as a Docker container, and deploy it via Kubernetes (Minikube) with a Pod, a Service, and basic resource limits.
**Scope, explicitly:** Docker, Dockerfile, containerization, Kubernetes, Minikube, Kubernetes Pod, Kubernetes Service, basic CPU/memory resource configuration, REST-based inference. **Explicitly out of scope**: OpenStack, Apache CloudStack, AWS/any paid cloud, CloudAnalyst, multiple replicas, horizontal/auto-scaling, load balancing, multiple VMs, distributed training. This matches the project's original CC scope -- the goal is to demonstrate meaningful cloud-native deployment of a real DL workload, not to reproduce a full production infrastructure.
**Prerequisite reading:** `DL_HANDOFF.md` -- the exact input/output contract, dependency list, and directory layout `predict.py` needs. Read that before starting here; this roadmap assumes it.

---

## Phase 0 -- Environment Setup (~30-45 min)

Install (all free, no account/paid tier required):
- **Docker Desktop** (includes the Docker Engine + CLI): https://www.docker.com/products/docker-desktop/
- **kubectl** (Kubernetes CLI): comes bundled with Docker Desktop on Windows/Mac if you enable Kubernetes in its settings, or install standalone.
- **Minikube** (runs a single-node Kubernetes cluster locally): https://minikube.sigs.k8s.io/docs/start/

Verify:
```bash
docker --version
kubectl version --client
minikube version
```

Start Minikube once to confirm it works before building anything:
```bash
minikube start
kubectl get nodes    # should show one node, status "Ready"
```

**Deliverable:** all three commands above run without error, `kubectl get nodes` shows one Ready node.

---

## Phase 1 -- Get the model checkpoint into the build context (~5 min, depends on transfer method)

`checkpoints/best_model/` (~470MB) is not in git -- see `DL_HANDOFF.md`'s "Getting the checkpoint" section for how it's transferred from the DL side. Place it at:

```
cc-deploy/
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
    patch_dnabert2.py
    paths.py
  requirements.txt
```

(`cc-deploy/` here is just a suggested folder name for the deployment build context -- adjust to whatever the actual working directory is. The key requirement is that `checkpoints/best_model/`, `src/predict.py`, `src/patch_dnabert2.py`, and `src/paths.py` all sit in the layout `paths.py`'s repo-root resolution expects: `checkpoints/` and `src/` as siblings.)

**Deliverable:** the directory structure above exists and `model.safetensors` is present (not an empty/placeholder file -- check its size, should be ~450MB+).

---

## Phase 2 -- FastAPI wrapper (~30-45 min)

`predict.py` already has a clean `predict(seq, tokenizer=None, model=None)` function -- this phase just exposes it over HTTP, loading the model once at startup rather than per-request (see `DL_HANDOFF.md`'s performance notes on why that matters).

Create `app.py` at the repo root (alongside `src/`):

```python
# app.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import load, predict as run_predict

model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer, model = load()
    model_state["tokenizer"] = tokenizer
    model_state["model"] = model
    yield
    model_state.clear()


app = FastAPI(title="GenePromoter Inference API", lifespan=lifespan)


class SequenceRequest(BaseModel):
    sequence: str = Field(..., description="Promoter DNA sequence, uppercase A/C/G/T", min_length=1)


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in model_state}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(req: SequenceRequest):
    result = run_predict(req.sequence, tokenizer=model_state["tokenizer"], model=model_state["model"])
    return result
```

Add to `requirements.txt` (or a separate `requirements-api.txt`, whichever the project's convention is): `fastapi`, `uvicorn[standard]`, `pydantic`.

**Test locally before containerizing** (catches bugs faster than debugging inside Docker):
```bash
pip install fastapi "uvicorn[standard]"
uvicorn app:app --host 0.0.0.0 --port 8000
```
In another terminal:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"sequence\": \"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\"}"
```

**Deliverable:** `/health` returns `{"status": "ok", "model_loaded": true}`, and `/predict` returns a `{"prediction": ..., "confidence": ...}` JSON body.

---

## Phase 3 -- Dockerfile (~30 min)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps some torch/transformers builds need for compiling wheels; harmless if unused
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Trim to what predict.py + the API actually need (see DL_HANDOFF.md) --
# torch/transformers/einops for inference, fastapi/uvicorn/pydantic for the API.
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY checkpoints/ ./checkpoints/
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Note on image size**: this image will be multi-GB (torch alone is typically 1-2GB, plus the 470MB checkpoint) -- that's expected and fine for this scope (no requirement to minimize image size here), not a sign something's wrong.

**Note on internet access at runtime**: `predict.py`'s `ensure_patched()` reaches `huggingface.co` on first call to fetch/patch DNABERT-2's base remote code (see `DL_HANDOFF.md`). The container needs outbound internet access for that first call to succeed -- if deploying somewhere fully air-gapped, that HuggingFace cache would need to be pre-populated and baked into the image instead (an enhancement beyond this roadmap's base scope, flag it if it becomes relevant).

Build and test the image standalone first, before bringing Kubernetes into it:
```bash
docker build -t genepromoter-api:latest .
docker run -p 8000:8000 genepromoter-api:latest
```
In another terminal, same curl checks as Phase 2.

**Deliverable:** `docker build` completes, `docker run` serves the same `/health` and `/predict` responses as the local FastAPI test.

---

## Phase 4 -- Get the image into Minikube (~10 min)

Minikube runs its own isolated Docker environment by default -- an image built with your regular `docker build` isn't automatically visible to it. Two options, either is fine for this scope:

**Option A -- build directly inside Minikube's Docker daemon:**
```bash
eval $(minikube docker-env)      # Windows PowerShell: & minikube -p minikube docker-env | Invoke-Expression
docker build -t genepromoter-api:latest .
```

**Option B -- load a pre-built image into Minikube:**
```bash
docker build -t genepromoter-api:latest .
minikube image load genepromoter-api:latest
```

**Deliverable:** `minikube image ls | grep genepromoter-api` (or `docker images` after `minikube docker-env`, for Option A) shows the image is present.

---

## Phase 5 -- Kubernetes manifests: Deployment + Service (~30 min)

A **Pod** is the smallest deployable unit (one running instance of the container); a **Deployment** manages Pods (here, just one replica -- no scaling per this project's scope); a **Service** gives the Pod(s) a stable network endpoint to be reached through.

`k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genepromoter-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: genepromoter-api
  template:
    metadata:
      labels:
        app: genepromoter-api
    spec:
      containers:
        - name: genepromoter-api
          image: genepromoter-api:latest
          imagePullPolicy: Never   # use the locally-loaded image, don't try to pull from a registry
          ports:
            - containerPort: 8000
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 10
```

`k8s/service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: genepromoter-service
spec:
  selector:
    app: genepromoter-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: NodePort
```

**Why these resource numbers:** 2Gi memory request / 4Gi limit gives headroom for torch + a 117M-parameter model loaded in memory plus request handling; 1-2 CPU cores is reasonable for single-request CPU inference (recall from `DL_HANDOFF.md`: CPU inference works fine for single predictions, no GPU needed for serving). Adjust up if `kubectl describe pod` shows OOM-kills or the Pod fails to become Ready within a reasonable time.

**`imagePullPolicy: Never`** is important -- without it, Kubernetes tries to pull the image from a remote registry (Docker Hub etc.) instead of using the one just loaded locally into Minikube, and the Pod will fail with `ImagePullBackOff`.

**Deliverable:** both YAML files exist and `kubectl apply --dry-run=client -f k8s/` reports no syntax errors.

---

## Phase 6 -- Deploy and test (~20 min)

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl get pods                 # watch until STATUS is Running and READY is 1/1
kubectl describe pod <pod-name>  # if it's stuck -- check Events at the bottom for the actual error
kubectl logs <pod-name>          # check for startup errors, including the model-loading step
```

Reach the service (Minikube-specific, since `NodePort` alone isn't directly reachable the way it would be on a real cluster):
```bash
minikube service genepromoter-service --url
```
This prints a URL (e.g. `http://192.168.49.2:31234`). Test against it:
```bash
curl http://192.168.49.2:31234/health
curl -X POST http://192.168.49.2:31234/predict -H "Content-Type: application/json" -d "{\"sequence\": \"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\"}"
```

**Deliverable:** both endpoints respond correctly through the full Kubernetes path (not just `docker run` directly) -- this is the actual "cloud-native deployment" proof point for the project.

---

## Phase 7 -- Cleanup / teardown (when done demoing)

```bash
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
minikube stop        # or `minikube delete` to fully remove the cluster
```

---

## Troubleshooting quick-reference

| Symptom | Likely cause | Check |
|---|---|---|
| `ImagePullBackOff` | Minikube can't see the image, or `imagePullPolicy` isn't `Never` | Re-run Phase 4, confirm `imagePullPolicy: Never` is in the deployment YAML |
| Pod stuck `Pending` | Resource requests exceed what Minikube's VM has available | `kubectl describe pod`, check Events; lower `requests` or increase Minikube's resources (`minikube start --cpus=4 --memory=8g`) |
| Pod `CrashLoopBackOff` | App error on startup -- likely a missing dependency or a path issue in the container | `kubectl logs <pod-name>`; common culprits: checkpoint not actually copied into the image (check `Dockerfile`'s `COPY checkpoints/`), or `src/patch_dnabert2.py`/`src/paths.py` not copied alongside `predict.py` |
| `/predict` times out or errors on first request | `ensure_patched()` needs outbound internet to reach `huggingface.co` and the Pod has no egress | Confirm Minikube's network allows outbound internet; test `curl https://huggingface.co` from inside the Pod (`kubectl exec -it <pod-name> -- bash`) |
| Predictions seem wrong/nonsensical for valid promoter sequences | Wrong decision threshold, or checkpoint files incomplete | Confirm `checkpoints/best_model/model.safetensors` is the full ~450MB+ file, not truncated; confirm `predict.py`'s `HIGH_THRESHOLD = 0.66` wasn't accidentally reverted |

---

## Deliverables checklist (what "CC side done" looks like)

- [ ] Phase 0: Docker, kubectl, Minikube installed and verified
- [ ] Phase 1: checkpoint + source files in the correct build-context layout
- [ ] Phase 2: FastAPI app serves `/health` and `/predict` locally (no Docker yet)
- [ ] Phase 3: same endpoints work from a standalone `docker run`
- [ ] Phase 4: image loaded into Minikube
- [ ] Phase 5: Deployment + Service YAML written, with resource requests/limits set
- [ ] Phase 6: endpoints reachable and correct through the full `kubectl apply` + `minikube service` path

If all boxes are checked, the CC side has a working, demonstrable, cloud-native deployment of the DL model -- matching the project's stated scope (Docker + Kubernetes Pod/Service + basic resource configuration), without over-building into territory (scaling, multi-VM, paid cloud) that was explicitly descoped from the start.
