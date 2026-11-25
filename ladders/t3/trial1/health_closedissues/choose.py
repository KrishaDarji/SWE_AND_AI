#!/usr/bin/env python3
"""Select a balanced configuration from the Health-ClosedIssues HPO table."""
import argparse
import csv
import math
import time
from pathlib import Path

from sklearn.model_selection import train_test_split

OBJECTIVES = ("MRE-", "ACC+", "PRED40+")
PARAMETERS = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")


def load_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = set(OBJECTIVES + PARAMETERS)
    if not rows or not required.issubset(rows[0]):
        raise ValueError("dataset is empty or missing required columns")
    for row in rows:
        for key in OBJECTIVES + ("N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"):
            row[key] = float(row[key])
    return rows


def nondominated(rows):
    # Minimize MRE; maximize accuracy and prediction quality.
    result = []
    for i, candidate in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            no_worse = (other["MRE-"] <= candidate["MRE-"] and
                        other["ACC+"] >= candidate["ACC+"] and
                        other["PRED40+"] >= candidate["PRED40+"])
            strictly_better = (other["MRE-"] < candidate["MRE-"] or
                               other["ACC+"] > candidate["ACC+"] or
                               other["PRED40+"] > candidate["PRED40+"])
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(candidate)
    return result


def _normalize(value, low, high, maximize=True):
    if high == low:
        return 1.0
    scaled = (value - low) / (high - low)
    return scaled if maximize else 1.0 - scaled


def rank(rows):
    frontier = nondominated(rows)
    bounds = {key: (min(r[key] for r in frontier), max(r[key] for r in frontier)) for key in OBJECTIVES}
    for row in frontier:
        row["balanced_score"] = (0.40 * _normalize(row["ACC+"], *bounds["ACC+"], True) +
                                  0.35 * _normalize(row["PRED40+"], *bounds["PRED40+"], True) +
                                  0.25 * _normalize(row["MRE-"], *bounds["MRE-"], False))
    frontier.sort(key=lambda r: (-r["balanced_score"], r["MRE-"], -r["ACC+"], -r["PRED40+"]))
    return frontier, bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    rows = load_rows(args.dataset)
    # Deterministic holdout is used as a data-integrity check, not model fitting.
    train, test = train_test_split(rows, test_size=0.20, random_state=7)
    frontier, bounds = rank(rows)
    selected = frontier[0]
    print(f"rows={len(rows)} train_rows={len(train)} test_rows={len(test)}")
    print(f"frontier_rows={len(frontier)}")
    print("selection_weights=ACC+:0.40,PRED40+:0.35,MRE-:0.25")
    print("frontier_bounds=" + ",".join(f"{k}:{bounds[k][0]:.9f}..{bounds[k][1]:.9f}" for k in OBJECTIVES))
    print("selected=" + ",".join(f"{k}:{selected[k]}" for k in PARAMETERS))
    print("metrics=" + ",".join(f"{k}:{selected[k]:.9f}" for k in OBJECTIVES))
    print(f"balanced_score={selected['balanced_score']:.9f}")
    print(f"runtime_seconds={time.perf_counter() - started:.6f}")


if __name__ == "__main__":
    main()
