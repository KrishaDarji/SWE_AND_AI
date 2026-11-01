#!/usr/bin/env python3
import argparse, json
from analyzer import choose
p = argparse.ArgumentParser(description="Choose a historical model configuration under a finite estimator cap.")
p.add_argument("dataset"); p.add_argument("--cap", type=float, default=100)
a = p.parse_args()
print(json.dumps(choose(a.dataset, a.cap), indent=2, sort_keys=True, allow_nan=False))
