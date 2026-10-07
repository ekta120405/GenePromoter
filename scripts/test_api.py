"""
Smoke test for the deployed API. Standard library only, works on Windows/Mac/Linux.

Usage:
  python scripts/test_api.py                          # defaults to http://localhost:8000
  python scripts/test_api.py http://192.168.49.2:30080  # URL from `minikube service genepromoter-service --url`

Runs /health, /info, then predicts all 10 real held-out test promoters in
static/samples.json and prints predicted vs true label + latency.
Screenshot the output for the report.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
SAMPLES = os.path.join(os.path.dirname(__file__), "..", "static", "samples.json")


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


print(f"Target: {BASE}\n")
print("GET /health ->", *call("/health"))
print("GET /info   ->", *call("/info"))

print("\nPOST /predict with an invalid sequence (expect 422):")
print("  ", call("/predict", {"sequence": "ACGTNNNN" * 10})[0])

print("\nPOST /predict on 10 held-out test-set promoters:")
print(f"  {'gene':<12}{'true':<6}{'predicted':<18}{'conf%':>6}{'ms':>9}  ok")
correct, t0 = 0, time.time()
for s in json.load(open(SAMPLES)):
    code, r = call("/predict", {"sequence": s["sequence"]})
    if code != 200:
        print(f"  {s['gene']:<12}ERROR {code}: {r}")
        continue
    ok = r["prediction"].startswith(s["true_label"])
    correct += ok
    print(f"  {s['gene']:<12}{s['true_label']:<6}{r['prediction']:<18}{r['confidence']:>6}{r['inference_ms']:>9}  {'Y' if ok else 'N'}")
print(f"\n{correct}/10 correct, total {time.time() - t0:.1f}s, served by pod {r.get('served_by', '?')}")
