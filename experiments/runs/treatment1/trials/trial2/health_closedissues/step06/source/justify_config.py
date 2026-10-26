#!/usr/bin/env python3
"""Recommend a model configuration from a closed-issues HPO result CSV."""

import argparse
import csv
import json
import math
import os
import sys


REQUIRED = ("MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
            "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
OBJECTIVES = (("MRE-", False), ("ACC+", True), ("PRED40+", True))


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(name not in reader.fieldnames for name in REQUIRED):
            raise ValueError("CSV is missing one or more required columns")
        rows = []
        for number, row in enumerate(reader, 2):
            try:
                parsed = dict(row)
                for name, _ in OBJECTIVES:
                    parsed[name] = float(row[name])
                for name in ("N_estimators", "Min_sample_leaves", "Max_depth"):
                    parsed[name] = int(float(row[name]))
                parsed["Min_impurity_decrease"] = float(row["Min_impurity_decrease"])
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid numeric value on CSV line %d" % number) from exc
            if any(not math.isfinite(parsed[name]) for name, _ in OBJECTIVES):
                raise ValueError("non-finite objective on CSV line %d" % number)
            rows.append(parsed)
    if not rows:
        raise ValueError("CSV contains no data rows")
    return rows


def desirability(value, low, high, maximize):
    if high == low:
        return 1.0
    fraction = (value - low) / (high - low)
    return fraction if maximize else 1.0 - fraction


def recommend(rows):
    bounds = {name: (min(row[name] for row in rows), max(row[name] for row in rows))
              for name, _ in OBJECTIVES}
    ranked = []
    for index, row in enumerate(rows):
        components = {name: desirability(row[name], *bounds[name], maximize)
                      for name, maximize in OBJECTIVES}
        score = sum(components.values()) / len(components)
        complexity = (row["N_estimators"], row["Max_depth"], row["Min_sample_leaves"])
        ranked.append((score, complexity, index, row, components))
    # Highest balanced desirability; ties prefer fewer trees, shallower depth,
    # and larger leaves, then stable input order.
    ranked.sort(key=lambda item: (-item[0], item[1][0], item[1][1], -item[1][2], item[2]))
    score, _, index, row, components = ranked[0]
    return {
        "configuration": {name: row[name] for name in REQUIRED[3:]},
        "score": score,
        "objective_desirability": components,
        "source_row": index + 2,
        "rows_considered": len(rows),
        "justification": (
            "This row maximizes the equal-weight mean of min-max normalized "
            "desirability: lower MRE and higher ACC/PRED40. The selected "
            "configuration is therefore balanced across all three observed "
            "validation objectives; tie-breaking favors a simpler model."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Justify a model configuration from HPO results")
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    result = recommend(read_rows(args.csv_path))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
