# CC Project Context (read first)

Cloud Computing mini project: containerized, Kubernetes based deployment of the GenePromoter DNABERT-2 gene expression model.
Group: Ekta Baphana (B013), Ishitaa Jain (B051), Khushi Kedia (B063), Tanu Lodha (B070).

## What this repo is
- DL side (already done, do not modify): `src/`, `notebooks/`, `data/`, `results/`. Fine tuned DNABERT-2 predicts HIGH/LOW liver gene expression from a 1000bp promoter. ~69% test accuracy, decision threshold 0.66.
- CC side (this branch, `cc-deployment`): `app.py`, `static/`, `Dockerfile`, `.dockerignore`, `requirements-api.txt`, `k8s/`, `scripts/test_api.py`.
- Reference docs: `DL_HANDOFF.md` (input/output contract), `CC_ROADMAP.md` (phase plan), `README.md` in this kit (run steps).

## Approved scope (from the submitted title doc)
Docker container, Kubernetes via Minikube, Pod, Service, cloud native deployment, REST API inference, resource provisioning.
Out of scope: paid cloud, scaling/replicas, load balancing, multi VM, OpenStack.

Title doc workflow, and where each step lives:
1. Trained DNABERT-2 model: `checkpoints/best_model/` (NOT in git, ~470MB, transferred separately)
2. REST API: `app.py` (FastAPI)
3. Docker image/container: `Dockerfile`
4. Kubernetes/Minikube: `k8s/`
5. Pod: created by `k8s/deployment.yaml`
6. Service: `k8s/service.yaml` (NodePort 30080)
7. DNA in, HIGH/LOW out: `/predict` endpoint, web page at `/`

## Status
- [x] CC code written; API logic tested against a stub model
- [x] Checkpoint obtained from DL side
- [x] Kit reviewed and fixed (see `CHANGELOG_CC.md`)
- [x] Local API run with real model (`uvicorn app:app`): 7/10, no segfault on Windows
- [x] Docker build + run: offline run passes, ~940 MiB, evidence in `evidence/`
- [ ] Minikube deploy + test through Service (Phase 2, see `RUNBOOK.md`)
- [ ] Demo screenshots
- [ ] Review docs: requirements, design doc, tools used, research paper abstract + lit review, "why CC / combined project" justification

## Device plan
Two phases, two laptops.

Phase 1, this laptop (8GB, ~5.9GB usable): steps 0 to 4 only (plain run, Docker build + run). No Minikube.
- `%UserProfile%\.wslconfig` caps the Docker VM at 4GB (copied from `wslconfig.txt`)
- Close browser and other apps during Docker steps
- Never run the uvicorn server and the Docker container at the same time
- Python 3.12 via `py -3.12` (3.13 is the default `python`)
- Use PowerShell; use `scripts/test_api.py` instead of curl (PowerShell curl is a different command)

Phase 2, friend's 16GB laptop: Minikube deploy, test, evidence. Follow `RUNBOOK.md`.
- Pod requests 500m CPU / 1500Mi, limits 2 CPU / 2500Mi (set in `k8s/deployment.yaml`)

## Never
- Commit `checkpoints/` (GitHub 100MB limit; already gitignored).
- Change `HIGH_THRESHOLD` or anything in `src/` without the DL side.

## Writing style for docs
Short, punchy copy. No dashes. No filler. Clean formatting.
