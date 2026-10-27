"""Domain objects for software-project history."""

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class Project:
    """One historical project, retaining raw categorical and numeric fields."""

    project_id: str
    year: Optional[int]
    kloc: Optional[float]
    effort: Optional[float]
    defects: Optional[float]
    months: Optional[float]
    attributes: Mapping[str, str]


@dataclass(frozen=True)
class Summary:
    """Aggregates and delivery-outcome statistics for a project collection."""

    project_count: int
    year_min: Optional[int]
    year_max: Optional[int]
    total_kloc: Optional[float]
    total_effort: Optional[float]
    total_defects: Optional[float]
    average_months: Optional[float]
    completed_projects: int

    def as_lines(self) -> list[str]:
        def number(value: Optional[float]) -> str:
            return "unavailable" if value is None else f"{value:.2f}"

        years = "unavailable" if self.year_min is None else f"{self.year_min}-{self.year_max}"
        return [
            f"Historical projects: {self.project_count}",
            f"Project years: {years}",
            f"Total size (KLOC): {number(self.total_kloc)}",
            f"Total effort: {number(self.total_effort)} person-months",
            f"Total reported defects: {number(self.total_defects)}",
            f"Average delivery duration: {number(self.average_months)} months",
            f"Projects with recorded delivery duration: {self.completed_projects}/{self.project_count}",
        ]
