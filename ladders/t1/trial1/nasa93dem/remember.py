#!/usr/bin/env python3
"""Summarize historical NASA93 project characteristics and effort measures."""
import argparse
import csv
import json
import math
from collections import Counter

OUTCOME_FIELDS = ("Kloc+", "Effort-", "Defects-", "MonthsX")
ID_FIELDS = ("idX", "centerX", "YearX")


def _number(value, field, row_number):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("row %d has invalid %s: %r" % (row_number, field, value)) from exc


def load_projects(path):
    """Load and validate all projects from a NASA93 CSV file."""
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        missing = [name for name in OUTCOME_FIELDS + ID_FIELDS if name not in reader.fieldnames]
        if missing:
            raise ValueError("CSV is missing required fields: " + ", ".join(missing))
        projects = []
        for row_number, row in enumerate(reader, 2):
            project = dict(row)
            for field in OUTCOME_FIELDS:
                project[field] = _number(row[field], field, row_number)
            for field in ID_FIELDS:
                if not row[field].strip():
                    raise ValueError("row %d has an empty %s" % (row_number, field))
            projects.append(project)
    if not projects:
        raise ValueError("CSV contains no projects")
    return projects


def _stats(values):
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((value - mean) ** 2 for value in ordered) / count
    middle = ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) / 2
    return {"count": count, "min": ordered[0], "max": ordered[-1],
            "mean": mean, "median": middle, "stdev": math.sqrt(variance)}


def summarize(projects):
    """Return deterministic characteristics and effort/outcome summaries."""
    characteristics = {}
    for field in ID_FIELDS:
        characteristics[field] = dict(sorted(Counter(row[field].strip() for row in projects).items()))
    # Driver fields are the categorical columns between YearX and the outcomes.
    driver_fields = [field for field in projects[0] if field not in ID_FIELDS + OUTCOME_FIELDS]
    characteristics["drivers"] = {
        field: dict(sorted(Counter(row[field].strip() for row in projects).items()))
        for field in driver_fields
    }
    outcomes = {field: _stats([row[field] for row in projects]) for field in OUTCOME_FIELDS}
    return {"project_count": len(projects), "characteristics": characteristics, "outcomes": outcomes}


def render_report(summary):
    lines = ["NASA93 historical project summary", "Projects: %d" % summary["project_count"], ""]
    lines.append("Characteristics:")
    for field, values in summary["characteristics"].items():
        if field == "drivers":
            continue
        lines.append("  %s: %s" % (field, ", ".join("%s=%s" % item for item in values.items())))
    lines.append("  driver levels:")
    for field, values in summary["characteristics"]["drivers"].items():
        lines.append("    %s: %s" % (field, ", ".join("%s=%s" % item for item in values.items())))
    lines.append("Effort measures and outcomes:")
    for field, values in summary["outcomes"].items():
        lines.append("  %s: mean=%.2f median=%.2f min=%.2f max=%.2f stdev=%.2f" %
                     (field, values["mean"], values["median"], values["min"], values["max"], values["stdev"]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    result = summarize(load_projects(args.csv_path))
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else render_report(result))


if __name__ == "__main__":
    main()
