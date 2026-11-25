"""Pure summary functions; no I/O or presentation policy."""

from collections import defaultdict
from typing import Iterable

from .model import DatasetSummary, Project

OUTCOMES = ("kloc", "effort", "defects", "months")


def _values(projects, field):
    return [getattr(project, field) for project in projects]


def summarize_projects(projects: Iterable[Project], dataset: str) -> DatasetSummary:
    projects = list(projects)
    if not projects:
        raise ValueError("cannot summarize an empty project collection")
    totals = {field: sum(_values(projects, field)) for field in OUTCOMES}
    averages = {field: totals[field] / len(projects) for field in OUTCOMES}
    minima, maxima = {}, {}
    for field in OUTCOMES:
        low = min(projects, key=lambda p: getattr(p, field))
        high = max(projects, key=lambda p: getattr(p, field))
        minima[field] = {"project_id": low.project_id, "value": getattr(low, field)}
        maxima[field] = {"project_id": high.project_id, "value": getattr(high, field)}
    grouped = defaultdict(list)
    for project in projects:
        grouped[project.year].append(project)
    by_year = []
    for year in sorted(grouped):
        cohort = grouped[year]
        by_year.append({"year": year, "project_count": len(cohort),
                        "averages": {f: sum(_values(cohort, f)) / len(cohort) for f in OUTCOMES}})
    years = sorted(grouped)
    return DatasetSummary(dataset, len(projects), [years[0], years[-1]], totals,
                          averages, minima, maxima, by_year)
