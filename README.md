# GenePromoter CC Deployment

Docker + Kubernetes (Minikube) deployment of the DNABERT-2 gene expression model from the GenePromoter repo.

```
DNA sequence -> Service (NodePort 30080) -> Pod -> container (FastAPI + uvicorn) -> DNABERT-2 -> HIGH/LOW + confidence
```

## What's in here

| File | Purpose |
|---|---|
| `app.py` | FastAPI REST API. Loads the model once at startup. `/predict`, `/health`, `/info`, `/docs`, and a web page at `/` |
| `static/index.html` | Demo web page. "Load real test-set promoter" fills in one of 10 held-out test sequences and shows true vs predicted label |
| `static/samples.json` | Those 10 real test promoters (5 HIGH, 5 LOW) |
| `requirements-api.txt` | Runtime deps only (no pandas/sklearn/matplotlib) |
| `Dockerfile` | CPU-only torch, bakes the HuggingFace cache in at build time so the Pod needs no internet |
| `k8s/deployment.yaml` | 1 replica, CPU/memory requests + limits, startup/readiness/liveness probes |
| `k8s/service.yaml` | NodePort Service on fixed port 30080 |
| `scripts/test_api.py` | Smoke test: health, info, invalid input, and 10 real predictions |

## Setup: merge into the GenePromoter repo

Copy everything in this folder into the **root** of the GenePromoter repo (next to `src/`), then put the checkpoint in place:

```
GenePromoter/
  app.py  Dockerfile  .dockerignore  requirements-api.txt
  static/  k8s/  scripts/
  src/predict.py  src/patch_dnabert2.py  src/paths.py     (already there)
  checkpoints/best_model/model.safetensors  (+ config.json, bert_layers.py, ...)   <- get from DL side
```

Check `model.safetensors` is ~450MB+, not a stub. `.gitignore` already excludes `checkpoints/`, so it won't get committed.

## Step 1. Install tools (once)

Docker Desktop, kubectl, Minikube. Verify:

```bash
docker --version
kubectl version --client
minikube version
```

## Step 2. Run the API locally, no Docker (optional but catches bugs fastest)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-api.txt
uvicorn app:app --port 8000
```

Open http://localhost:8000 and http://localhost:8000/docs. Then `python scripts/test_api.py`.

## Step 3. Build and run the Docker image

```bash
docker build -t genepromoter-api:1.0 .
docker run -p 8000:8000 genepromoter-api:1.0
python scripts/test_api.py
```

First build takes 10 to 20 min (torch download + model cache bake). Rebuilds after code edits are fast because of layer caching.

## Step 4. Start Minikube and load the image

Full Phase 2 steps (16GB laptop, build inside Minikube, evidence, troubleshooting): see `RUNBOOK.md`.

```bash
minikube start --driver=docker --cpus=4 --memory=6144
minikube image load genepromoter-api:1.0
minikube image ls | grep genepromoter       # PowerShell: | findstr genepromoter
```

## Step 5. Deploy

```bash
kubectl apply -f k8s/
kubectl get pods -w          # wait for READY 1/1, Ctrl+C to stop watching
kubectl get svc
```

## Step 6. Test through Kubernetes

```bash
minikube service genepromoter-service --url
python scripts/test_api.py <that URL>
```

Open the URL in a browser for the web page. On Windows with the Docker driver, keep the `minikube service` terminal open while testing.

## Demo commands worth screenshotting

```bash
kubectl get all                          # Deployment, ReplicaSet, Pod, Service
kubectl describe pod <pod-name>          # resource requests/limits, probes, events
kubectl logs <pod-name>                  # "[startup] model loaded in Xs"
kubectl top pod                          # live CPU/memory (run `minikube addons enable metrics-server` first)
kubectl delete pod <pod-name>            # self-healing: watch a new Pod get created automatically
kubectl get pods -w
minikube dashboard                       # visual cluster view
```

## Teardown

```bash
kubectl delete -f k8s/
minikube stop
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch==2.14.0` not found during build | Remove the version pin on the torch line in the Dockerfile |
| Build fails at the "bake" step | Checkpoint missing or incomplete in `checkpoints/best_model/`, or no internet during build |
| Container fails on startup with an offline/cache error | Delete the `ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` lines and rebuild (Pod will then need internet on first start) |
| `ImagePullBackOff` / `ErrImageNeverPull` | Image not loaded into Minikube; redo Step 4, tag must match `genepromoter-api:1.0` |
| Pod stuck `Pending` | Not enough resources; `minikube delete` then `minikube start --driver=docker --cpus=4 --memory=6144` |
| `OOMKilled` / `CrashLoopBackOff` | `kubectl logs <pod>`; raise `limits.memory` to `3Gi` (needs `minikube start --memory=3584`) |
