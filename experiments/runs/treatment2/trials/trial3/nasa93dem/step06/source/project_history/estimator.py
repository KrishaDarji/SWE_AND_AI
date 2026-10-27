"""Effort estimation for screening a proposed software project."""

from statistics import median
from typing import Iterable, Optional

from .model import Project


def estimate_effort(projects: Iterable[Project], proposed_kloc: float) -> Optional[float]:
    """Estimate person-months using the median observed effort per KLOC."""
    if proposed_kloc <= 0:
        raise ValueError("proposed KLOC must be greater than zero")
    productivity = [
        item.effort / item.kloc
        for item in projects
        if item.effort is not None and item.kloc is not None and item.kloc > 0
    ]
    return proposed_kloc * median(productivity) if productivity else None

