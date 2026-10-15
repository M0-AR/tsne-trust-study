import argparse, json, sys
sys.path.insert(0, "src")
from tsne_trust.experiments import run_all

ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true")
ap.add_argument("--out", default="results/results.json")
a = ap.parse_args()
R = run_all(fast=a.fast, out=a.out)
print(json.dumps({k: R[k] for k in list(R)[:12]}, indent=2))
print("wrote", a.out)
