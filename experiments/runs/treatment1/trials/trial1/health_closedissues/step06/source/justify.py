#!/usr/bin/env python3
"""Recommend and justify a model configuration from historical measurements."""
import argparse
import csv
import math
import statistics
from pathlib import Path

METRICS = ("MRE-", "ACC+", "PRED40+")
PARAMETERS = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")

def load_records(path):
    records = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                values = {key: float(row[key]) for key in METRICS}
                if any(not math.isfinite(value) or value < 0 for value in values.values()): continue
                values.update({key: row[key].strip() for key in PARAMETERS})
                values["score"] = values["ACC+"] + values["PRED40+"] - values["MRE-"]
                records.append(values)
            except (KeyError, TypeError, ValueError):
                continue
    if not records:
        raise ValueError("dataset contains no valid records")
    return records

def analyze(records, top_fraction=0.10):
    ranked = sorted(records, key=lambda row: row["score"], reverse=True)
    top = ranked[:max(1, math.ceil(len(ranked) * top_fraction))]
    recommendation = {key: statistics.multimode([row[key] for row in top])[0] for key in PARAMETERS}
    return {"records": len(records), "top_count": len(top), "best": ranked[0],
            "top_mean": statistics.mean(row["score"] for row in top),
            "overall_mean": statistics.mean(row["score"] for row in records),
            "overall_median": statistics.median(row["score"] for row in records),
            "recommendation": recommendation}

def render(result):
    best = result["best"]
    config = ", ".join(f"{key}={value}" for key, value in result["recommendation"].items())
    lines = ["Historical model-configuration justification",
             f"Records used: {result['records']} (top {result['top_count']} by composite score)",
             "Composite score: ACC+ + PRED40+ - MRE- (higher is better)",
             f"Historical score: mean {result['overall_mean']:.4f}; median {result['overall_median']:.4f}",
             f"Top-group mean score: {result['top_mean']:.4f}",
             f"Recommended configuration (modal values among top group): {config}",
             f"Best observed evidence: score {best['score']:.4f}, MRE- {best['MRE-']:.4f}, ACC+ {best['ACC+']:.4f}, PRED40+ {best['PRED40+']:.4f}",
             "Use this as a starting point, then confirm with held-out validation; historical observations do not establish causality."]
    return "\n".join(lines)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    args = parser.parse_args(argv)
    if not 0 < args.top_fraction <= 1: parser.error("--top-fraction must be in (0, 1]")
    print(render(analyze(load_records(args.dataset), args.top_fraction)))

if __name__ == "__main__":
    main()
