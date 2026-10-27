"""Find a historical project whose effort rate is unusually inconsistent."""

from dataclasses import dataclass
from statistics import median
from typing import Iterable, List

from .model import Project


@dataclass(frozen=True)
class InconsistentProject:
    project_id: str
    effort_per_kloc: float
    historical_median: float
    robust_score: float
    reason: str

    def to_dict(self):
        return self.__dict__.copy()


def spot_inconsistent(projects: Iterable[Project], top: int = 1) -> List[InconsistentProject]:
    """Return the most unusual positive-KLOC records by robust effort rate.

    The modified z-score uses median absolute deviation (MAD), so a few extreme
    projects do not pull the historical center toward themselves.  When MAD is
    zero, exact deviations are used as a deterministic fallback.
    """
    projects = [p for p in projects if p.kloc > 0]
    if not projects:
        raise ValueError("historical projects must contain positive KLOC")
    if top <= 0:
        raise ValueError("top must be positive")
    rates = [p.effort / p.kloc for p in projects]
    center = median(rates)
    mad = median(abs(rate - center) for rate in rates)
    scale = 1.4826 * mad
    scored = []
    for project, rate in zip(projects, rates):
        score = abs(rate - center) / scale if scale else abs(rate - center)
        direction = "higher" if rate > center else "lower"
        scored.append(InconsistentProject(
            project.project_id, rate, center, score,
            "%s effort per KLOC than the historical median" % direction))
    scored.sort(key=lambda item: (-item.robust_score, item.project_id))
    return scored[:top]
