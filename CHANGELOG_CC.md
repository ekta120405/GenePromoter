# CC Kit Change Log

What changed in the deployment kit after review, and why. `src/` and the DL contract are untouched.

## Dockerfile

| Change | Why |
|---|---|
| Model bake step moved **before** `COPY app.py` and `static/` | Docker rebuilds every layer after a changed one. Before, any API edit re-downloaded the base model and reloaded it. Now an `app.py` edit rebuilds two tiny layers |
| Base DNABERT-2 weights (~470MB) deleted after the bake | The fine-tuned weights come from `checkpoints/best_model`. The base repo is only used for the tokenizer and code files. Baked cache went from ~470MB to 0.5MB |
| `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` | Matches the Pod's 2 CPU limit. Stops torch from spawning more threads than the CPU quota allows |
| Kept `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | Pod never contacts Hugging Face. Proven with `docker run --network none` |

## app.py

| Change | Why |
|---|---|
| `/predict` adds `label`, `probability_high`, `threshold` | Clear HIGH/LOW plus the raw P(HIGH) the 0.66 threshold is applied to. Original `prediction` and `confidence` kept, so the web page, test script and `DL_HANDOFF.md` contract still hold |
| Empty input returns "sequence is empty" | Was reported as "must contain only A, C, G, T", which is misleading |
| Startup log shows torch thread count | Evidence that the thread cap is active: `[startup] model loaded in 4.4s (torch threads: 2)` |

## k8s/deployment.yaml

| Change | Why |
|---|---|
| `timeoutSeconds: 5` on all three probes | Default is 1s. While the CPU is busy predicting, `/health` can be slow to answer and Kubernetes would restart a healthy Pod |

Already correct in the kit, kept as is: Deployment with 1 replica (not a bare Pod, so deleting the Pod recreates it), `imagePullPolicy: Never`, startup probe with a 5 min budget, requests 500m / 1500Mi, limits 2 CPU / 2500Mi, NodePort 30080.

Measured memory is ~940 MiB, so the 1500Mi request and 2500Mi limit leave safe headroom.

## Other

| Change | Why |
|---|---|
| `.dockerignore` adds `evidence/`, `.claude/`, `.vscode/`, `wslconfig.txt` | Smaller build context, nothing irrelevant in the image |
| `RUNBOOK.md` added | Phase 2 steps for the 16GB laptop: Minikube, deploy, test, evidence, troubleshooting |
| `evidence/` added | Phase 1 proof: startup logs, test outputs, `docker stats`, image size, offline run |
| `README.md` Minikube flags | Point to `RUNBOOK.md` and the 16GB laptop sizing |

## Phase 1 results

| Gate | Result |
|---|---|
| Plain run (Windows, Python 3.12) | Pass. No segfault: the Windows crash affects training only, not inference |
| Docker build | Pass, 4m18s, ~1.8GB of layers |
| Docker run, no network | Pass, model loads in ~4s, predictions identical to plain run |
| Test script | 7/10 held-out promoters correct in both runs |
| Memory | 908 MiB idle, 942 MiB under load |
