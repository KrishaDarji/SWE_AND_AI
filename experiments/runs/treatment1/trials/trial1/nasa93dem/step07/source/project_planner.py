#!/usr/bin/env python3
"""Select the least-effort NASA93 project that fits an effort budget."""

import argparse
import csv
import json
from pathlib import Path


def load_projects(path):
    """Load numeric planning fields from a NASA93 CSV file."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        projects = []
        for row in csv.DictReader(handle):
            try:
                projects.append({
                    "id": int(row["idX"]),
                    "effort": float(row["Effort-"]),
                    "defects": float(row["Defects-"]),
                    "months": float(row["MonthsX"]),
                    "kloc": float(row["Kloc+"]),
                })
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid project row: %r" % row) from exc
    if not projects:
        raise ValueError("the project file contains no usable rows")
    return projects


def choose_project(projects, budget):
    """Return the best project whose estimated effort is at most budget.

    Ranking is effort ascending, then defects, duration, size, and id.  The
    explicit tie-breakers make repeated runs deterministic and avoid treating
    a larger project as inherently better when the requirement only supplies
    an effort constraint.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    feasible = [project for project in projects if project["effort"] <= budget]
    if not feasible:
        raise ValueError("no project satisfies the effort budget")
    return min(feasible, key=lambda p: (
        p["effort"], p["defects"], p["months"], p["kloc"], p["id"]
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--budget", type=float, required=True,
                        help="maximum estimated effort")
    args = parser.parse_args(argv)
    result = choose_project(load_projects(args.csv_path), args.budget)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
