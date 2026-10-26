#!/usr/bin/env python3
"""Summarize historical NASA93 software projects and delivery outcomes."""

import argparse
import csv
import json
import statistics
from pathlib import Path


NUMERIC_FIELDS = ("Kloc+", "Effort-", "Defects-", "MonthsX")


def load_projects(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset contains no projects")
    missing = [field for field in NUMERIC_FIELDS if field not in rows[0]]
    if missing:
        raise ValueError("missing required fields: " + ", ".join(missing))
    for row in rows:
        for field in NUMERIC_FIELDS:
            row[field] = float(row[field].strip())
        row["YearX"] = int(row["YearX"].strip())
    return rows


def _stats(rows, field):
    values = [row[field] for row in rows]
    return {"min": min(values), "median": statistics.median(values), "max": max(values), "mean": statistics.mean(values)}


def summarize(rows):
    by_year = {}
    for row in rows:
        by_year.setdefault(str(row["YearX"]), []).append(row)
    return {
        "project_count": len(rows),
        "year_range": [min(row["YearX"] for row in rows), max(row["YearX"] for row in rows)],
        "outcomes": {field: _stats(rows, field) for field in NUMERIC_FIELDS},
        "projects_by_year": {year: len(group) for year, group in sorted(by_year.items())},
        "delivery_interpretation": "MonthsX is treated as delivery duration; Effort-, Defects-, and Kloc+ are reported as recorded outcomes.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(load_projects(args.dataset)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
