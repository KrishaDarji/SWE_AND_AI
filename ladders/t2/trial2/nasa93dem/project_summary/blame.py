"""Rank project factors associated with excessive historical effort."""

from dataclasses import dataclass
from math import sqrt
from statistics import median
from typing import Iterable, List

from .model import Project


@dataclass(frozen=True)
class FactorAssociation:
    factor: str
    value: str
    projects: int
    excessive_projects: int
    excessive_rate: float
    baseline_rate: float
    lift: float
    mean_effort: float
    score: float

    def to_dict(self):
        return self.__dict__.copy()


@dataclass(frozen=True)
class BlameReport:
    effort_threshold: float
    excessive_projects: int
    project_count: int
    factors: List[FactorAssociation]

    def to_dict(self):
        return {"effort_threshold": self.effort_threshold,
                "excessive_projects": self.excessive_projects,
                "project_count": self.project_count,
                "factors": [factor.to_dict() for factor in self.factors]}


def blame_factors(projects: Iterable[Project], threshold=None, top=10,
                  minimum_group_size=3) -> BlameReport:
    """Rank categorical factor values by excessive-effort rate lift.

    Excessive effort defaults to effort at or above the dataset median.  The
    score combines absolute rate lift with group size, so tiny groups do not
    dominate the ranking.  This is association, not causal attribution.
    """
    projects = list(projects)
    if not projects:
        raise ValueError("cannot analyze an empty project collection")
    if top <= 0 or minimum_group_size <= 0:
        raise ValueError("top and minimum group size must be positive")
    effort_threshold = median(p.effort for p in projects) if threshold is None else float(threshold)
    excessive = [p.effort >= effort_threshold for p in projects]
    baseline = sum(excessive) / len(projects)
    factors = []
    names = sorted({name for p in projects for name in p.attributes
                    if name not in {"idX", "YearX", "Kloc+", "Effort-", "Defects-", "MonthsX"}})
    for name in names:
        values = sorted({p.attributes.get(name, "") for p in projects})
        for value in values:
            group = [p for p in projects if p.attributes.get(name, "") == value]
            if len(group) < minimum_group_size:
                continue
            count = sum(p.effort >= effort_threshold for p in group)
            rate = count / len(group)
            lift = rate - baseline
            score = abs(lift) * sqrt(len(group))
            factors.append(FactorAssociation(name, value, len(group), count,
                                              rate, baseline, lift,
                                              sum(p.effort for p in group) / len(group), score))
    factors.sort(key=lambda f: (-f.score, -f.lift, f.factor, f.value))
    return BlameReport(effort_threshold, sum(excessive), len(projects), factors[:top])
