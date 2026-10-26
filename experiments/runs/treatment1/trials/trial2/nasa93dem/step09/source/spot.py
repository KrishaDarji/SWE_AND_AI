#!/usr/bin/env python3
"""Spot records whose numeric outcomes are unusual for this data set."""

import argparse
import csv
import math
import statistics
import sys


OUTCOME_FIELDS = ("Effort-", "Defects-", "MonthsX")


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def median_and_mad(values):
    center = statistics.median(values)
    mad = statistics.median(abs(value - center) for value in values)
    return center, mad


def score_rows(rows):
    """Return rows sorted by a robust unusualness score, highest first.

    Scores use log(outcome / KLOC), which compares scale-adjusted effort,
    defects, and schedule.  Median/MAD scoring is resistant to a few outliers.
    """
    if not rows:
        return []
    observations = {field: [] for field in OUTCOME_FIELDS}
    for row in rows:
        kloc = float(row["Kloc+"])
        if kloc <= 0:
            raise ValueError("Kloc+ must be positive")
        for field in OUTCOME_FIELDS:
            observations[field].append(math.log(float(row[field]) / kloc))
    baselines = {field: median_and_mad(values) for field, values in observations.items()}
    scored = []
    for index, row in enumerate(rows):
        components = []
        for field in OUTCOME_FIELDS:
            value = observations[field][index]
            center, mad = baselines[field]
            scale = 1.4826 * mad
            components.append(abs(value - center) / scale if scale else 0.0)
        scored.append((max(components), sum(components) / len(components), row))
    return sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spot an inconsistent NASA93 project record")
    parser.add_argument("csv_path")
    parser.add_argument("--top", type=int, default=1)
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be at least 1")
    ranked = score_rows(read_rows(args.csv_path))
    print("id,score,mean_score,signals")
    for maximum, mean, row in ranked[:args.top]:
        kloc = float(row["Kloc+"])
        signals = ",".join(field for field in OUTCOME_FIELDS
                            if abs(math.log(float(row[field]) / kloc) -
                                   statistics.median(math.log(float(r[field]) / float(r["Kloc+"]))
                                                      for r in read_rows(args.csv_path))) > 0)
        print(f"{row['idX']},{maximum:.3f},{mean:.3f},{signals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
