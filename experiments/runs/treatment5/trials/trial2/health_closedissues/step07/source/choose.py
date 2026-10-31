#!/usr/bin/env python3
"""Select the best observed configuration under an estimator cap."""
import csv
import math
import sys

FIELDS = ("N_estimators", "criterion", "Min_sample_leaves",
          "Min_impurity_decrease", "Max_depth")
OUTCOMES = ("MRE-", "ACC+", "PRED40+")

def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def choose(records, cap):
    """Return a result dictionary, preserving 1-based CSV record identifiers."""
    if isinstance(cap, bool) or not finite(cap) or float(cap) < 0:
        raise ValueError("cap must be a finite nonnegative number")
    cap = float(cap)
    total = len(records)
    feasible = []
    for record_id, row in enumerate(records, 1):
        if any(not str(row.get(k, "")).strip() for k in FIELDS + OUTCOMES):
            continue
        if not row["criterion"].strip():
            continue
        if any(not finite(row[k]) for k in OUTCOMES + ("N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")):
            continue
        if float(row["N_estimators"]) < 0 or float(row["N_estimators"]) > cap:
            continue
        feasible.append((record_id, row))
    if not feasible:
        raise LookupError("no feasible records")
    record_id, row = min(feasible, key=lambda item: (
        float(item[1]["MRE-"]), -float(item[1]["ACC+"]),
        -float(item[1]["PRED40+"]), item[0]))
    return {"cap": cap, "total": total, "feasible": len(feasible),
            "record_id": record_id,
            "configuration": {k: row[k] for k in FIELDS},
            "outcomes": {k: row[k] for k in OUTCOMES}}

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))

def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: choose.py DATASET CAP")
    result = choose(read_csv(argv[1]), float(argv[2]))
    print("cap={cap:g} feasible={feasible} total={total} record_id={record_id}".format(**result))
    print("configuration=" + ",".join(f"{k}={v}" for k, v in result["configuration"].items()))
    print("outcomes=" + ",".join(f"{k}={v}" for k, v in result["outcomes"].items()))

if __name__ == "__main__":
    main(sys.argv)
