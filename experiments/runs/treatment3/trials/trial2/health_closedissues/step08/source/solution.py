#!/usr/bin/env python3
"""Evaluate one-parameter fixes using the supplied Health HPO results."""
import csv
import os
import sys
from collections import defaultdict

PARAMETERS = ["N_estimators", "criterion", "Min_sample_leaves",
              "Min_impurity_decrease", "Max_depth"]
METRICS = ["MRE-", "ACC+", "PRED40+"]


def load_dataset(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = set(METRICS + PARAMETERS)
    if not rows or not required.issubset(rows[0]):
        raise ValueError("missing required columns or empty dataset")
    for row in rows:
        for metric in METRICS:
            row[metric] = float(row[metric])
        for parameter in PARAMETERS:
            if not row[parameter]:
                raise ValueError("empty parameter value")
    return rows


def signature(row, omitted=None):
    return tuple(row[p] for p in PARAMETERS if p != omitted)


def one_change_candidates(rows, baseline):
    index = defaultdict(list)
    for row in rows:
        for parameter in PARAMETERS:
            index[signature(row, parameter)].append(row)
    candidates = []
    seen = set()
    for parameter in PARAMETERS:
        for row in index[signature(baseline, parameter)]:
            changed = tuple(p for p in PARAMETERS if row[p] != baseline[p])
            key = tuple(row[p] for p in PARAMETERS)
            if len(changed) == 1 and key not in seen:
                seen.add(key)
                candidates.append(row)
    return candidates


def choose_baseline(rows):
    # Deterministic, data-only choice: maximize the number of measured
    # one-parameter comparisons, then prefer lower error and higher quality.
    counts = []
    index = defaultdict(int)
    for row in rows:
        for parameter in PARAMETERS:
            index[signature(row, parameter)] += 1
    for position, row in enumerate(rows):
        neighbors = sum(index[signature(row, p)] - 1 for p in PARAMETERS)
        counts.append((neighbors, -row["MRE-"], row["ACC+"] + row["PRED40+"], -position, row))
    return max(counts)[-1]


def evaluate(path):
    rows = load_dataset(path)
    baseline = choose_baseline(rows)
    candidates = one_change_candidates(rows, baseline)
    # Primary objective: lower MRE. Tie-break by total prediction quality.
    best = min(candidates, key=lambda r: (r["MRE-"], -(r["ACC+"] + r["PRED40+"]))) if candidates else None
    changed = [p for p in PARAMETERS if best and best[p] != baseline[p]]
    result = {"rows": len(rows), "baseline": baseline, "candidate_count": len(candidates),
              "best": best, "changed_parameter": changed[0] if len(changed) == 1 else "",
              "error_gain": baseline["MRE-"] - best["MRE-"] if best else 0.0,
              "acc_change": best["ACC+"] - baseline["ACC+"] if best else 0.0,
              "pred40_change": best["PRED40+"] - baseline["PRED40+"] if best else 0.0}
    return result


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: solution.py DATASET.csv")
    result = evaluate(argv[1])
    b, best = result["baseline"], result["best"]
    print("rows=%d candidates=%d" % (result["rows"], result["candidate_count"]))
    print("baseline: MRE=%.12f ACC=%.12f PRED40=%.12f config=%s" %
          (b["MRE-"], b["ACC+"], b["PRED40+"], {p: b[p] for p in PARAMETERS}))
    if best:
        print("revised:  MRE=%.12f ACC=%.12f PRED40=%.12f changed=%s" %
              (best["MRE-"], best["ACC+"], best["PRED40+"], result["changed_parameter"]))
        print("gain: error_reduction=%.12f acc_change=%+.12f pred40_change=%+.12f" %
              (result["error_gain"], result["acc_change"], result["pred40_change"]))
        print("recommendation: change %s from %s to %s" %
              (result["changed_parameter"], b[result["changed_parameter"]], best[result["changed_parameter"]]))
    else:
        print("recommendation: no measured one-parameter improvement found")


if __name__ == "__main__":
    main(sys.argv)
