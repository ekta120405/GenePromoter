# GenePromoter Minikube Runbook (8GB laptop)

Deploy the GenePromoter API to local Kubernetes on the same 8GB laptop used for Phase 1, test it through the Service, and capture demo evidence.
Phase 1 (plain run + Docker) already passed. Results are in `evidence/`.

Use **PowerShell** for every command.

```
DNA sequence -> Service (NodePort 30080) -> Pod -> container (FastAPI) -> DNABERT-2 -> HIGH/LOW
```

## Memory and disk budget

| Item | Size |
|---|---|
| Docker VM cap (`%UserProfile%\.wslconfig`) | 4GB |
| Minikube node (inside that VM) | 3GB |
| Kubernetes system Pods | ~0.7GB |
| GenePromoter Pod (measured) | ~0.94GB |
| Disk needed on C: for Minikube + image | ~6GB free |

Rules on this laptop:
- Close the browser, VS Code extensions you don't need, and other apps before starting Minikube.
- Never run `uvicorn` or a plain `docker run` container while Minikube is up.
- Open the browser only for the web page demo, one tab.

## What to expect

| Thing | Value measured in Phase 1 |
|---|---|
| Image size | ~1.8GB of layers |
| Model load at startup | ~4s |
| Container memory | ~940 MiB |
| Prediction time | ~0.5 to 1s per sequence |
| Test script score | 7/10 held-out promoters correct (matches ~69% test accuracy) |

---

## 1. Tools (once)

Docker Desktop and kubectl are already installed (kubectl ships with Docker Desktop). Install Minikube:

```powershell
winget install -e --id Kubernetes.minikube
```

Open a **new** PowerShell window, then check:

```powershell
docker version          # must show a Server section
kubectl version --client
minikube version
```

## 2. Free disk space

Check free space on C:. You need ~6GB.

```powershell
Get-PSDrive C
docker system df
```

If it is short, clear the Docker build cache (safe; it only slows the next rebuild):

```powershell
docker builder prune -f
```

Do **not** delete the `genepromoter-api:1.0` image yet. Step 4 copies it into Minikube.

## 3. Start Minikube

```powershell
minikube start --driver=docker --cpus=2 --memory=3072
kubectl get nodes
```

Pass: one node, `STATUS Ready`. First start downloads ~1GB and takes a few minutes.

## 4. Load the image into Minikube

Load the exact image that passed Phase 1 testing:

```powershell
minikube image load genepromoter-api:1.0
minikube image ls | findstr genepromoter
```

Pass: the last line shows `docker.io/library/genepromoter-api:1.0`. Takes a few minutes.

Optional, once it is loaded, to win back ~2.6GB of disk:

```powershell
docker rmi genepromoter-api:1.0
```

You can always rebuild it with `docker build -t genepromoter-api:1.0 .` (~5 min, needs internet).

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
.\.venv\Scripts\python.exe scripts\test_api.py http://127.0.0.1:54321
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
.\.venv\Scripts\python.exe scripts\test_api.py http://127.0.0.1:54321
```

The `served by pod` name at the bottom is now different. Same URL, new Pod.

Skip `minikube dashboard` and the metrics-server addon on this laptop. They cost memory the Pod needs.

To save text copies too: `kubectl get all | Out-File -Encoding utf8 evidence\phase2_get_all.txt` (same pattern for the others).

## 8. Teardown

```powershell
kubectl delete -f k8s/
minikube stop
```

`minikube stop` frees the RAM. `minikube delete` also frees the disk (cluster + loaded image).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `minikube start` fails on memory | Docker VM has less than 3GB free | Quit apps, `wsl --shutdown`, restart Docker Desktop, retry. Last resort: `--memory=2800` |
| Laptop freezes | Host RAM exhausted | Lower `memory=` in `%UserProfile%\.wslconfig` to 3.5GB, `wsl --shutdown`, use `minikube start --memory=2800` |
| `ErrImageNeverPull` | Image is not inside Minikube (`imagePullPolicy: Never`) | Redo step 4. Tag must be exactly `genepromoter-api:1.0` |
| `ErrImagePull` / `ImagePullBackOff` | Manifest was changed to pull from a registry | Keep `imagePullPolicy: Never` in `k8s/deployment.yaml` |
| `Pending` | Node can't fit the 1500Mi memory request | `kubectl describe pod $POD`, read Events. Lower `requests.memory` to `1Gi` in `k8s/deployment.yaml` (Pod uses ~940 MiB), `kubectl apply -f k8s/` |
| `CrashLoopBackOff` | App crashes at startup | `kubectl logs $POD --previous`. Usually a missing or partial checkpoint at build time. Rebuild, redo step 4, then `kubectl rollout restart deployment/genepromoter-api` |
| `OOMKilled` in `describe pod` | Hit the 2500Mi limit | Unlikely at ~940 MiB. Check `kubectl logs $POD --previous` for a memory spike |
| `Running` but `READY 0/1` | Probes failing | `kubectl describe pod $POD`, check Events for probe failures. Model load takes ~4s, so look at `kubectl logs $POD` for an error |
| `minikube image load` fails with no space | C: is full | Step 2, then `minikube delete` and start again |
| `minikube service --url` window seems frozen | Normal. It is holding the tunnel open | Leave it running, use the URL from another window |
| Test script times out | Tunnel window was closed | Rerun `minikube service genepromoter-service --url`, use the new URL |
| Code changed, Pod still runs old code | Same tag, old image cached | `docker build -t genepromoter-api:1.0 .`, `minikube image load genepromoter-api:1.0`, `kubectl rollout restart deployment/genepromoter-api` |
