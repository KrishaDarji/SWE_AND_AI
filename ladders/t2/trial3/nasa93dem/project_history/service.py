"""Use-case logic kept independent from CSV input and command-line output."""

from typing import Iterable, Optional

from .model import Project, Summary


def _sum(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def summarize_projects(projects: Iterable[Project]) -> Summary:
    """Compute delivery-oriented aggregates from any iterable of projects."""
    items = list(projects)
    years = [item.year for item in items if item.year is not None]
    return Summary(
        project_count=len(items),
        year_min=min(years) if years else None,
        year_max=max(years) if years else None,
        total_kloc=_sum(item.kloc for item in items),
        total_effort=_sum(item.effort for item in items),
        total_defects=_sum(item.defects for item in items),
        average_months=_mean(item.months for item in items),
        completed_projects=sum(item.months is not None for item in items),
    )
