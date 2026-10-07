# GenePromoter Phase 2 Runbook: Minikube on the 16GB laptop

Deploy the GenePromoter API to local Kubernetes, test it through the Service, and capture demo evidence.
Phase 1 (plain run + Docker) already passed on the 8GB laptop. Results are in `evidence/`.

Use **PowerShell** for every command. Copy and paste them as written.

```
DNA sequence -> Service (NodePort 30080) -> Pod -> container (FastAPI) -> DNABERT-2 -> HIGH/LOW
```

## What to expect

| Thing | Value measured in Phase 1 |
|---|---|
| Image size | ~1.8GB of layers |
| Build time | ~5 min (first build, needs internet) |
| Model load at startup | ~4s |
| Container memory | ~940 MiB |
| Prediction time | ~0.5 to 1s per sequence |
| Test script score | 7/10 held-out promoters correct (matches ~69% test accuracy) |

---

## 1. Install tools (once)

```powershell
winget install -e --id Docker.DockerDesktop
winget install -e --id Kubernetes.kubectl
winget install -e --id Kubernetes.minikube
winget install -e --id Git.Git
winget install -e --id Python.Python.3.12
```

Restart the laptop after Docker Desktop installs. Start Docker Desktop and wait for **Engine running**.
Open a **new** PowerShell window, then check:

```powershell
docker version          # must show a Server section
kubectl version --client
minikube version
git --version
python --version
```

All five must work before you continue.

## 2. Get the code and the checkpoint

```powershell
cd $HOME\Desktop
git clone https://github.com/ekta120405/GenePromoter
cd GenePromoter
git checkout cc-deployment
```

The checkpoint is **not** in git (~470MB). Copy the `best_model` folder you were given to `checkpoints\best_model\`:

```powershell
dir checkpoints\best_model
```

Pass: you see `config.json`, `model.safetensors` (~468MB, not a few KB), `tokenizer.json`, `tokenizer_config.json`, `bert_layers.py`, `bert_padding.py`, `configuration_bert.py`.

## 3. Start Minikube

Close the browser and heavy apps first.

```powershell
minikube start --driver=docker --cpus=4 --memory=6144
kubectl get nodes
```

Pass: one node, `STATUS Ready`.

If it says the memory is more than Docker has, open Docker Desktop, Settings, Resources, and raise memory to 8GB, or use `--memory=5120`.

## 4. Build the image inside Minikube

**Recommended:** build straight into Minikube's own Docker. No 2GB image copy, normal build output, layer caching works.

```powershell
& minikube -p minikube docker-env --shell powershell | Invoke-Expression
docker build -t genepromoter-api:1.0 .
minikube image ls | findstr genepromoter
```

Pass: the last line shows `docker.io/library/genepromoter-api:1.0`.

The `docker-env` line only affects **this** terminal. If you open a new window and rebuild, run it again first, or the image lands in Docker Desktop instead of Minikube.

Fallback if the build inside Minikube fails: build on the host, then copy it in (slower).

```powershell
minikube docker-env --shell powershell --unset | Invoke-Expression
docker build -t genepromoter-api:1.0 .
minikube image load genepromoter-api:1.0
```

## 5. Deploy

```powershell
kubectl apply -f k8s/
kubectl rollout status deployment/genepromoter-api --timeout=300s
kubectl get pods,svc
```

Pass: rollout says `successfully rolled out`, Pod shows `READY 1/1` and `STATUS Running`, Service `genepromoter-service` shows `80:30080/TCP`.

## 6. Test through the Service

Open a **second** PowerShell window and run this. It holds a tunnel open, so **leave that window open** while testing:

```powershell
minikube service genepromoter-service --url
```

It prints a URL like `http://127.0.0.1:54321`. Back in the **first** window:

```powershell
python scripts\test_api.py http://127.0.0.1:54321
```

Pass: `/health` 200, invalid sequence 422, `7/10 correct`, `served by pod genepromoter-api-...`.

Then open the same URL in a browser. Click **Load real test-set promoter**, then **Predict**. `/docs` on the same URL shows the Swagger UI.

## 7. Evidence for the report

Take a screenshot after each command.

```powershell
$POD = kubectl get pods -l app=genepromoter-api -o jsonpath='{.items[0].metadata.name}'
kubectl get all
kubectl describe pod $POD          # requests/limits, probes, events
kubectl logs $POD                  # shows "[startup] model loaded in Xs"
```

**Self healing demo.** Use two windows.

Window A:
```powershell
kubectl get pods -w
```

Window B:
```powershell
kubectl delete pod $POD
```

Window A shows the old Pod `Terminating` and a new one going `ContainerCreating`, `Running`, then `1/1`. Ctrl+C to stop watching. Then prove the Service still works with a new Pod name:

```powershell
python scripts\test_api.py http://127.0.0.1:54321
```

The `served by pod` name at the bottom is now different. Same URL, new Pod.

Optional live usage:
```powershell
minikube addons enable metrics-server
kubectl top pod                    # wait ~1 min after enabling
minikube dashboard                 # visual view, opens a browser
```

To save text copies too: `kubectl get all | Out-File -Encoding utf8 evidence\phase2_get_all.txt` (same pattern for the others).

## 8. Teardown

```powershell
kubectl delete -f k8s/
minikube stop
```

`minikube delete` wipes the cluster and the image completely if you need the disk space back.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ErrImageNeverPull` | Image is not inside Minikube (`imagePullPolicy: Never`) | Redo step 4 in a terminal where `docker-env` was run. Tag must be exactly `genepromoter-api:1.0` |
| `ErrImagePull` / `ImagePullBackOff` | Manifest was changed to pull from a registry | Keep `imagePullPolicy: Never` in `k8s/deployment.yaml` |
| `Pending` | Not enough memory or CPU in Minikube | `kubectl describe pod $POD`, read Events. Then `minikube delete`, restart with more `--memory` |
| `CrashLoopBackOff` | App crashes at startup | `kubectl logs $POD --previous`. Usually a missing or partial `checkpoints\best_model` at build time. Fix it and rebuild (step 4), then `kubectl rollout restart deployment/genepromoter-api` |
| `OOMKilled` in `describe pod` | Hit the 2500Mi limit | Raise `limits.memory` to `3Gi` in `k8s/deployment.yaml`, `kubectl apply -f k8s/` |
| `Running` but `READY 0/1` | Probes failing | `kubectl describe pod $POD`, check Events for `Startup probe failed` / `Readiness probe failed`. Startup probe allows 5 min; model load takes ~4s, so look at `kubectl logs $POD` for an error |
| Build fails at the `model cache baked` step | No internet during build, or checkpoint missing | Check internet and step 2, rebuild |
| `minikube service --url` window seems frozen | Normal. It is holding the tunnel open | Leave it running, use the URL from another window |
| Test script times out | Tunnel window was closed | Rerun `minikube service genepromoter-service --url`, use the new URL |
| Code changed, Pod still runs old code | Same tag, old image cached | Rebuild (step 4), then `kubectl rollout restart deployment/genepromoter-api` |
