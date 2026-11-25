#!/usr/bin/env python3
import argparse, json
from analyzer import analyze
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Summarize model-configuration quality measurements.")
    p.add_argument("dataset"); args = p.parse_args()
    print(json.dumps(analyze(args.dataset), indent=2, sort_keys=True, allow_nan=False))
