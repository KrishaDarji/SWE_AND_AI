"""Robustly spot a historical project that is inconsistent with its peers."""

from statistics import median
from typing import Iterable, Optional

from .model import Project


def spot_inconsistent_project(projects: Iterable[Project], project_id: Optional[str] = None,
                              *, threshold: float = 3.5) -> dict[str, object]:
    """Return the most unusual effort/KLOC record using a modified z-score.

    The comparison is descriptive: a record is unusual when its productivity
    differs from the historical median by more than ``threshold`` robust
    standard deviations (MAD scaled by 1.4826).  The selected record may be
    supplied by id; otherwise the most unusual usable record is returned.
    """
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")
    items = [item for item in projects
             if item.kloc is not None and item.kloc > 0 and item.effort is not None]
    if not items:
        return {"project": None, "score": None, "median_productivity": None,
                "mad": None, "inconsistent": False, "reason": "insufficient data"}
    rates = [item.effort / item.kloc for item in items]
    center = median(rates)
    deviations = [abs(rate - center) for rate in rates]
    mad = median(deviations)
    if mad == 0:
        scores = [0.0 if rate == center else float("inf") for rate in rates]
    else:
        scores = [0.6745 * abs(rate - center) / mad for rate in rates]
    if project_id is None:
        index = max(range(len(items)), key=lambda position: (scores[position], items[position].project_id))
    else:
        matches = [position for position, item in enumerate(items) if item.project_id == project_id]
        if not matches:
            return {"project": None, "score": None, "median_productivity": center,
                    "mad": mad, "inconsistent": False, "reason": "project not found or unusable"}
        index = matches[0]
    item = items[index]
    score = scores[index]
    return {"project": item, "score": score, "median_productivity": center,
            "mad": mad, "inconsistent": score > threshold,
            "reason": "large productivity departure from historical median" if score > threshold
            else "within historical productivity variation"}
