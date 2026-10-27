"""Simple historical-baseline comparisons for effort estimates."""

from statistics import median
from typing import Iterable, Optional

from .estimator import estimate_effort
from .model import Project


def historical_baseline(projects: Iterable[Project]) -> Optional[float]:
    """Return the median observed effort, independent of project size."""
    efforts = [item.effort for item in projects if item.effort is not None]
    return median(efforts) if efforts else None


def compare_estimators(projects: Iterable[Project], proposed_kloc: float,
                       actual_effort: float) -> dict[str, Optional[float]]:
    """Compare the current size-aware estimator with the historical baseline."""
    if proposed_kloc <= 0:
        raise ValueError("proposed KLOC must be greater than zero")
    if actual_effort < 0:
        raise ValueError("actual effort cannot be negative")
    items = list(projects)
    estimate = estimate_effort(items, proposed_kloc)
    baseline = historical_baseline(items)
    return {
        "estimate": estimate,
        "baseline": baseline,
        "estimate_absolute_error": abs(estimate - actual_effort) if estimate is not None else None,
        "baseline_absolute_error": abs(baseline - actual_effort) if baseline is not None else None,
    }
