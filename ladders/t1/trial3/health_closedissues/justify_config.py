#!/usr/bin/env python3
"""Recommend and justify a random-forest configuration from experiment results."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

REQUIRED = ("MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
            "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")


def load_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(name not in rows[0] for name in REQUIRED):
        raise ValueError("CSV is empty or missing a required column")
    for row in rows:
        for name in REQUIRED[:3]:
            row[name] = float(row[name])
        for name in REQUIRED[3:]:
            row[name] = float(row[name]) if name == "Min_impurity_decrease" else int(row[name]) if name != "criterion" else row[name]
    return rows


def recommend(rows):
    # Equal-weight normalized utility: lower error and higher accuracy/coverage.
    def span(name):
        values = [r[name] for r in rows]
        return min(values), max(values)
    ranges = {name: span(name) for name in REQUIRED[:3]}
    def utility(row):
        mre = (ranges["MRE-"][1] - row["MRE-"]) / max(ranges["MRE-"][1] - ranges["MRE-"][0], 1e-12)
        acc = (row["ACC+"] - ranges["ACC+"][0]) / max(ranges["ACC+"][1] - ranges["ACC+"][0], 1e-12)
        pred = (row["PRED40+"] - ranges["PRED40+"][0]) / max(ranges["PRED40+"][1] - ranges["PRED40+"][0], 1e-12)
        return (mre + acc + pred) / 3.0
    ranked = sorted(((utility(row), row) for row in rows), key=lambda pair: pair[0], reverse=True)
    score, row = ranked[0]
    return {
        "configuration": {name: row[name] for name in REQUIRED[3:]},
        "score": score,
        "metrics": {name: row[name] for name in REQUIRED[:3]},
        "dataset_rows": len(rows),
        "metric_summary": {name: {"mean": statistics.fmean(r[name] for r in rows), "min": min(r[name] for r in rows), "max": max(r[name] for r in rows)} for name in REQUIRED[:3]},
        "justification": "This row maximizes the equal-weight normalized utility across lower MRE, higher ACC, and higher PRED40. The recommendation is therefore a balanced configuration rather than one optimized for a single metric."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args()
    print(json.dumps(recommend(load_rows(args.csv_path)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
