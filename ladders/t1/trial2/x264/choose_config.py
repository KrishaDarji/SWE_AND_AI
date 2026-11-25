#!/usr/bin/env python3
"""Choose the fastest feasible X264 configuration from a CSV dataset."""
import argparse
import csv
import json
import statistics
from pathlib import Path

OBJECTIVE = "Performance-"


def load_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    if OBJECTIVE not in rows[0]:
        raise ValueError("dataset lacks Performance- column")
    for row in rows:
        try:
            row[OBJECTIVE] = float(row[OBJECTIVE])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Performance- value") from exc
    return rows


def choose(path, max_performance=None):
    rows = load_rows(path)
    if max_performance is None:
        max_performance = statistics.median(row[OBJECTIVE] for row in rows)
    feasible = [row for row in rows if row[OBJECTIVE] <= max_performance]
    if not feasible:
        raise ValueError("quality constraint excludes every configuration")
    best = min(feasible, key=lambda row: (row[OBJECTIVE], tuple(row.items())))
    return {"constraint": max_performance, "feasible_count": len(feasible),
            "total_count": len(rows), "objective": OBJECTIVE, "best": best}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-performance", type=float,
                        help="maximum allowed Performance-; default is dataset median")
    args = parser.parse_args()
    print(json.dumps(choose(args.dataset, args.max_performance), sort_keys=True))


if __name__ == "__main__":
    main()
