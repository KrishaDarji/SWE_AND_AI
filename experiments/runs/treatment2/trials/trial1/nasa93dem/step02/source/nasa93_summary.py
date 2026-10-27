#!/usr/bin/env python3
"""Reusable historical-project summary for nasa93dem.csv (stdlib only)."""

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


OUTCOME_FIELDS = ("Kloc+", "Effort-", "Defects-", "MonthsX")


@dataclass(frozen=True)
class OutcomeSummary:
    field: str
    count: int
    minimum: float
    median: float
    mean: float
    maximum: float


def _percentile(values, fraction):
    """Linear-interpolated percentile using only the standard library."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _finite_number(value):
    try:
        number = float(value.strip())
    except (AttributeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_projects(path):
    """Load rows and validate the columns needed by later skills."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        missing = [name for name in OUTCOME_FIELDS if name not in reader.fieldnames]
        if missing:
            raise ValueError("missing outcome columns: " + ", ".join(missing))
        projects = []
        for line_number, row in enumerate(reader, start=2):
            parsed = {name: _finite_number(row.get(name)) for name in OUTCOME_FIELDS}
            if any(value is None for value in parsed.values()):
                raise ValueError("invalid outcome value on CSV line %d" % line_number)
            parsed["id"] = row.get("idX", "").strip() or str(line_number - 1)
            projects.append(parsed)
    if not projects:
        raise ValueError("CSV contains no projects")
    return projects


def _stats(values, field):
    return OutcomeSummary(field, len(values), min(values), statistics.median(values),
                           statistics.fmean(values), max(values))


def summarize_projects(path):
    """Return a JSON-serializable summary and row-level delivery indicators."""
    projects = load_projects(path)
    summaries = [_stats([project[field] for project in projects], field)
                 for field in OUTCOME_FIELDS]
    enriched = []
    for project in projects:
        item = dict(project)
        item["effort_per_kloc"] = project["Effort-"] / project["Kloc+"] if project["Kloc+"] else None
        item["defects_per_kloc"] = project["Defects-"] / project["Kloc+"] if project["Kloc+"] else None
        item["months_per_kloc"] = project["MonthsX"] / project["Kloc+"] if project["Kloc+"] else None
        enriched.append(item)
    return {"dataset": str(path), "project_count": len(projects),
            "outcomes": [asdict(summary) for summary in summaries],
            "projects": enriched}


def estimate_effort(path, kloc, budget=None):
    """Estimate effort for a proposed project from historical effort/KLOC.

    The median historical effort intensity is the point estimate.  The 25th
    and 75th percentiles form a planning range; approval requires a supplied
    budget to cover the upper planning estimate.
    """
    if not math.isfinite(float(kloc)) or float(kloc) <= 0:
        raise ValueError("kloc must be a positive finite number")
    if budget is not None and (not math.isfinite(float(budget)) or float(budget) < 0):
        raise ValueError("budget must be a non-negative finite number")
    projects = load_projects(path)
    intensities = [p["Effort-"] / p["Kloc+"] for p in projects if p["Kloc+"] > 0]
    median_rate = statistics.median(intensities)
    low_rate = _percentile(intensities, 0.25)
    high_rate = _percentile(intensities, 0.75)
    result = {
        "dataset": str(path), "kloc": float(kloc),
        "historical_projects": len(projects),
        "effort": float(kloc) * median_rate,
        "planning_low": float(kloc) * low_rate,
        "planning_high": float(kloc) * high_rate,
        "effort_per_kloc_median": median_rate,
        "budget": None if budget is None else float(budget),
        "approval": "needs_budget_review" if budget is None else
                    ("approve" if float(budget) >= float(kloc) * high_rate else "reject"),
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--estimate-kloc", type=float)
    parser.add_argument("--budget", type=float)
    args = parser.parse_args(argv)
    result = (estimate_effort(args.csv_path, args.estimate_kloc, args.budget)
              if args.estimate_kloc is not None else summarize_projects(args.csv_path))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
