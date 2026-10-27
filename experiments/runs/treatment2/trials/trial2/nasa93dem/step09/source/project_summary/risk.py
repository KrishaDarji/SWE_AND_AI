"""Recommend an observed process change that lowers effort risk cheaply."""

from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable, List

from .model import Project


@dataclass(frozen=True)
class ProcessChange:
    factor: str
    from_value: str
    to_value: str
    projects: int
    baseline_risk: float
    estimated_risk: float
    risk_reduction: float
    change_cost: float

    def to_dict(self):
        return self.__dict__.copy()


@dataclass(frozen=True)
class RiskRecommendation:
    baseline_risk: float
    recommendation: ProcessChange
    candidates: List[ProcessChange]

    def to_dict(self):
        return {"baseline_risk": self.baseline_risk,
                "recommendation": self.recommendation.to_dict(),
                "candidates": [candidate.to_dict() for candidate in self.candidates]}


def recommend_process_change(projects: Iterable[Project], top=10,
                             minimum_group_size=3) -> RiskRecommendation:
    """Choose the least costly observed categorical change that lowers risk.

    Risk is median effort per KLOC.  Change cost is the candidate group's mean
    effort, a transparent proxy for implementation cost available in this data.
    The comparison is associative, not causal, and only uses groups large
    enough to avoid one-project recommendations.
    """
    projects = list(projects)
    if not projects:
        raise ValueError("cannot recommend from an empty project collection")
    if top <= 0 or minimum_group_size <= 0:
        raise ValueError("top and minimum group size must be positive")
    rates = [p.effort / p.kloc for p in projects if p.kloc > 0]
    if not rates:
        raise ValueError("historical projects must contain positive KLOC")
    baseline = median(rates)
    excluded = {"idX", "YearX", "Kloc+", "Effort-", "Defects-", "MonthsX"}
    candidates = []
    names = sorted({name for p in projects for name in p.attributes if name not in excluded})
    for name in names:
        values = sorted({p.attributes.get(name, "") for p in projects})
        for value in values:
            group = [p for p in projects if p.attributes.get(name, "") == value and p.kloc > 0]
            if len(group) < minimum_group_size:
                continue
            risk = median(p.effort / p.kloc for p in group)
            reduction = baseline - risk
            if reduction <= 0:
                continue
            candidates.append(ProcessChange(name, "any other observed value", value,
                                            len(group), baseline, risk, reduction,
                                            mean(p.effort for p in group)))
    if not candidates:
        raise ValueError("no process change reduces estimated risk")
    candidates.sort(key=lambda c: (c.change_cost, -c.risk_reduction, c.factor, c.to_value))
    return RiskRecommendation(baseline, candidates[0], candidates[:top])
