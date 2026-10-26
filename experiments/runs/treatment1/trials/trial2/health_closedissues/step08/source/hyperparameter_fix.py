#!/usr/bin/env python3
"""Suggest the smallest one-hyperparameter change that lowers MRE."""

import argparse
import csv
import sys


ERROR_COLUMN = "MRE-"
HYPERPARAMETERS = ("N_estimators", "criterion", "Min_sample_leaves",
                   "Min_impurity_decrease", "Max_depth")


def read_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {ERROR_COLUMN, *HYPERPARAMETERS}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError("missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError("dataset has no data rows")
    return rows


def numeric(value):
    return float(value)


def change_distance(before, after, parameter):
    if parameter == "criterion":
        return 1.0
    return abs(numeric(after) - numeric(before))


def suggest(rows):
    baseline = rows[0]
    baseline_error = numeric(baseline[ERROR_COLUMN])
    candidates = []
    for index, row in enumerate(rows[1:], start=2):
        changed = [name for name in HYPERPARAMETERS
                   if row[name] != baseline[name]]
        if len(changed) != 1 or numeric(row[ERROR_COLUMN]) >= baseline_error:
            continue
        parameter = changed[0]
        key = (change_distance(baseline[parameter], row[parameter], parameter),
               numeric(row[ERROR_COLUMN]), index)
        candidates.append((key, parameter, row))
    if not candidates:
        return None
    _, parameter, row = min(candidates, key=lambda item: item[0])
    return {
        "baseline_error": baseline_error,
        "candidate_error": numeric(row[ERROR_COLUMN]),
        "parameter": parameter,
        "before": baseline[parameter],
        "after": row[parameter],
        "row": row,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="CSV containing evaluated configurations")
    args = parser.parse_args(argv)
    result = suggest(read_rows(args.dataset))
    if result is None:
        print("No improving one-hyperparameter change found.")
        return 0
    print("Suggested change: {parameter}: {before} -> {after}".format(**result))
    print("MRE: {baseline_error:.12g} -> {candidate_error:.12g}".format(**result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("error: " + str(error), file=sys.stderr)
        raise SystemExit(2)
