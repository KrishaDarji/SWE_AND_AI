"""CSV loading with a narrow, explicit schema boundary."""

import csv
from pathlib import Path
from typing import List

from .model import Project


def _number(row, name: str) -> float:
    value = row[name].strip()
    if not value:
        raise ValueError("empty numeric field: %s" % name)
    return float(value)


def load_projects(path: str) -> List[Project]:
    """Load projects from nasa93dem-compatible CSV data."""
    projects = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"idX", "YearX", "Kloc+", "Effort-", "Defects-", "MonthsX"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("missing required columns: %s" % ", ".join(sorted(missing)))
        for row in reader:
            projects.append(Project(
                project_id=row["idX"].strip(), year=int(row["YearX"]),
                kloc=_number(row, "Kloc+"), effort=_number(row, "Effort-"),
                defects=_number(row, "Defects-"), months=_number(row, "MonthsX"),
                attributes={k: (v or "").strip() for k, v in row.items()},
            ))
    return projects
