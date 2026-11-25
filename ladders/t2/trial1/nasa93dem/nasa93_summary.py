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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    text = json.dumps(summarize_projects(args.csv_path), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
