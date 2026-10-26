#!/usr/bin/env python3
"""Summarize historical NASA project delivery outcomes using the stdlib only."""

import argparse
import csv
import statistics
from collections import Counter, defaultdict


REQUIRED = ("YearX", "Kloc+", "Effort-", "Defects-", "MonthsX")


def load_projects(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CSV contains no project records")
    missing = [name for name in REQUIRED if name not in rows[0]]
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing))
    projects = []
    for row in rows:
        try:
            projects.append({
                "year": int(row["YearX"]),
                "kloc": float(row["Kloc+"]),
                "effort": float(row["Effort-"]),
                "defects": float(row["Defects-"]),
                "months": float(row["MonthsX"]),
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid numeric project row: " + repr(row)) from exc
    return projects


def _stats(projects):
    def mean(field):
        return statistics.fmean(p[field] for p in projects)
    return {
        "count": len(projects),
        "kloc_mean": mean("kloc"), "kloc_median": statistics.median(p["kloc"] for p in projects),
        "effort_mean": mean("effort"), "effort_median": statistics.median(p["effort"] for p in projects),
        "defects_mean": mean("defects"), "defects_median": statistics.median(p["defects"] for p in projects),
        "months_mean": mean("months"), "months_median": statistics.median(p["months"] for p in projects),
    }


def summarize(projects):
    if not projects:
        raise ValueError("cannot summarize an empty project list")
    years = Counter(p["year"] for p in projects)
    by_year = {year: _stats([p for p in projects if p["year"] == year]) for year in sorted(years)}
    durations = Counter("short (<=12 months)" if p["months"] <= 12 else
                        "medium (12-24 months)" if p["months"] <= 24 else
                        "long (>24 months)" for p in projects)
    return {"overall": _stats(projects), "year_counts": dict(sorted(years.items())),
            "by_year": by_year, "duration_counts": dict(sorted(durations.items()))}


def _line(label, value):
    return f"{label}: {value:.2f}"


def render(summary):
    overall = summary["overall"]
    lines = ["NASA93 historical project delivery summary", "", f"Projects analyzed: {overall['count']}",
             "", "Overall outcomes (mean; median in parentheses):",
             f"- Size: {_line('KLOC', overall['kloc_mean'])} ({overall['kloc_median']:.2f})",
             f"- Effort: {_line('person-months', overall['effort_mean'])} ({overall['effort_median']:.2f})",
             f"- Defects: {_line('count', overall['defects_mean'])} ({overall['defects_median']:.2f})",
             f"- Delivery time: {_line('months', overall['months_mean'])} ({overall['months_median']:.2f})", "",
             "Projects by historical year: " + ", ".join(f"{y}: {n}" for y, n in summary["year_counts"].items()),
             "Delivery-duration bands: " + ", ".join(f"{k}: {v}" for k, v in summary["duration_counts"].items()),
             "", "Year-level project counts and mean delivery months:"]
    for year, stats in summary["by_year"].items():
        lines.append(f"- {year}: {stats['count']} projects, {stats['months_mean']:.2f} months")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    print(render(summarize(load_projects(args.csv_path))), end="")


if __name__ == "__main__":
    main()
