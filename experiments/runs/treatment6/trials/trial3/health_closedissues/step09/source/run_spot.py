#!/usr/bin/env python3
import argparse, json
from analyzer import spot

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Screen unusual marginal model-quality outcomes.")
    p.add_argument("dataset")
    p.add_argument("--threshold", type=float, default=3.5)
    p.add_argument("--min-sample", type=int, default=3)
    a = p.parse_args()
    result = spot(a.dataset, a.threshold, a.min_sample)
    result["flagged_examples"] = result.pop("flagged_rows")[:10]
    result["flagged_examples_note"] = "First 10 flagged rows shown; spot() retains the complete flagged_rows list for programmatic use."
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
