"""Data models used by the project summarizer."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class Project:
    """One historical project, retaining raw fields and typed outcomes."""

    project_id: str
    year: int
    kloc: float
    effort: float
    defects: float
    months: float
    attributes: Dict[str, str]


@dataclass(frozen=True)
class DatasetSummary:
    """Serializable summary designed to support later reporting skills."""

    dataset: str
    project_count: int
    year_range: List[int]
    totals: Dict[str, float]
    averages: Dict[str, float]
    minima: Dict[str, Dict[str, Any]]
    maxima: Dict[str, Dict[str, Any]]
    by_year: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "project_count": self.project_count,
            "year_range": self.year_range,
            "totals": self.totals,
            "averages": self.averages,
            "minima": self.minima,
            "maxima": self.maxima,
            "by_year": self.by_year,
        }
