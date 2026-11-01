#!/usr/bin/env python3
import argparse, json
from analyzer import blame

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rank marginal configuration associations with MRE-.")
    p.add_argument("dataset")
    print(json.dumps(blame(p.parse_args().dataset), indent=2, sort_keys=False, allow_nan=False))
