"""Historical effort estimation for proposal screening."""

from dataclasses import dataclass
from statistics import median
from typing import Iterable, List

from .model import Project


@dataclass(frozen=True)
class EffortEstimate:
    """A transparent estimate and screening range for a proposed project."""

    proposed_kloc: float
    estimated_effort: float
    lower_effort: float
    upper_effort: float
    historical_projects: int
    method: str

    def to_dict(self):
        return {
            "proposed_kloc": self.proposed_kloc,
            "estimated_effort": self.estimated_effort,
            "lower_effort": self.lower_effort,
            "upper_effort": self.upper_effort,
            "historical_projects": self.historical_projects,
            "method": self.method,
        }


def estimate_effort(projects: Iterable[Project], proposed_kloc: float,
                    neighbors: int = 7) -> EffortEstimate:
    """Estimate effort from the median effort/KLOC of nearby historical cases."""
    projects = list(projects)
    if not projects:
        raise ValueError("cannot estimate from an empty project collection")
    if proposed_kloc <= 0:
        raise ValueError("proposed KLOC must be positive")
    if neighbors <= 0:
        raise ValueError("neighbors must be positive")
    ranked = sorted(projects, key=lambda p: abs(p.kloc - proposed_kloc))
    selected = ranked[:min(neighbors, len(ranked))]
    rates = [p.effort / p.kloc for p in selected if p.kloc > 0]
    if not rates:
        raise ValueError("historical projects must contain positive KLOC")
    center = median(rates)
    low = min(rates) * proposed_kloc
    high = max(rates) * proposed_kloc
    return EffortEstimate(proposed_kloc, center * proposed_kloc, low, high,
                          len(selected), "median effort/KLOC of nearest historical projects")
