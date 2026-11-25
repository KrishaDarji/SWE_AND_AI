"""Compare the nearest-neighbor estimator with a historical-rate baseline."""

from dataclasses import dataclass
from statistics import median
from typing import Iterable

from .model import Project


@dataclass(frozen=True)
class ComparisonResult:
    """Leave-one-out error report for two estimators."""

    projects_evaluated: int
    neighbors: int
    metric: str
    estimator_mape: float
    baseline_mape: float
    estimator_better: bool

    def to_dict(self):
        return {
            "projects_evaluated": self.projects_evaluated,
            "neighbors": self.neighbors,
            "metric": self.metric,
            "estimator_mape": self.estimator_mape,
            "baseline_mape": self.baseline_mape,
            "estimator_better": self.estimator_better,
        }


def compare_estimators(projects: Iterable[Project], neighbors: int = 7) -> ComparisonResult:
    """Evaluate nearest historical neighbors against an all-history median-rate baseline.

    Each project is held out before both predictions are made. MAPE is averaged
    over the same positive-effort rows for both methods.
    """
    projects = list(projects)
    if len(projects) < 2:
        raise ValueError("at least two projects are required")
    if neighbors <= 0:
        raise ValueError("neighbors must be positive")
    estimator_errors = []
    baseline_errors = []
    for index, actual in enumerate(projects):
        training = projects[:index] + projects[index + 1:]
        if actual.effort <= 0 or actual.kloc <= 0:
            continue
        ranked = sorted(training, key=lambda p: abs(p.kloc - actual.kloc))
        selected = [p for p in ranked[:min(neighbors, len(ranked))] if p.kloc > 0]
        if not selected:
            continue
        local_rate = median(p.effort / p.kloc for p in selected)
        global_rate = median(p.effort / p.kloc for p in training if p.kloc > 0)
        estimator_errors.append(abs(local_rate * actual.kloc - actual.effort) / actual.effort)
        baseline_errors.append(abs(global_rate * actual.kloc - actual.effort) / actual.effort)
    if not estimator_errors:
        raise ValueError("no valid positive-effort projects to evaluate")
    estimator_mape = 100.0 * sum(estimator_errors) / len(estimator_errors)
    baseline_mape = 100.0 * sum(baseline_errors) / len(baseline_errors)
    return ComparisonResult(len(estimator_errors), neighbors, "MAPE (%)",
                            estimator_mape, baseline_mape,
                            estimator_mape < baseline_mape)
