#!/usr/bin/env python3
"""Choose a balanced machine-learning configuration from a CSV file."""
import argparse
import csv
import json
from pathlib import Path

REQUIRED = ("MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
            "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")


def choose(path):
    """Return the best row and its score using fixed, documented weights."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(k not in reader.fieldnames for k in REQUIRED):
            raise ValueError("CSV is missing one or more required columns")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no configurations")

    def scored(row):
        acc = float(row["ACC+"])
        pred = float(row["PRED40+"])
        mre = float(row["MRE-"])
        if not all(0.0 <= x <= 1.0 for x in (acc, pred, mre)):
            raise ValueError("objective values must be in [0, 1]")
        score = 0.4 * acc + 0.4 * pred + 0.2 * (1.0 - mre)
        return score

    # max() preserves the first row on exact score ties, making selection stable.
    best = max(rows, key=scored)
    return best, scored(best)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    row, score = choose(args.csv_path)
    print(json.dumps({"score": round(score, 12), "configuration": row}, sort_keys=True))


if __name__ == "__main__":
    main()
