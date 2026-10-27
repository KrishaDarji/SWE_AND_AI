"""Choose historical project settings that maximize delivered value per effort."""

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List

from .model import Project


@dataclass(frozen=True)
class SettingRecommendation:
    settings: dict
    projects: int
    mean_kloc: float
    mean_effort: float
    value_per_effort: float
    feasible: bool

    def to_dict(self):
        return self.__dict__.copy()


@dataclass(frozen=True)
class OptimizationResult:
    budget: float
    value_metric: str
    settings_columns: List[str]
    recommendations: List[SettingRecommendation]
    selected: SettingRecommendation

    def to_dict(self):
        return {"budget": self.budget, "value_metric": self.value_metric,
                "settings_columns": self.settings_columns,
                "recommendations": [r.to_dict() for r in self.recommendations],
                "selected": self.selected.to_dict()}


def choose_settings(projects: Iterable[Project], budget: float,
                    settings_columns=None, top=10, minimum_group_size=2,
                    value_metric="kloc") -> OptimizationResult:
    """Rank observed setting combinations by mean delivered value per effort.

    KLOC is the default delivered-value proxy; effort is the budget/cost proxy.
    Only combinations whose mean historical effort fits the budget are eligible.
    """
    projects = list(projects)
    if not projects:
        raise ValueError("cannot optimize an empty project collection")
    if budget <= 0 or top <= 0 or minimum_group_size <= 0:
        raise ValueError("budget, top, and minimum group size must be positive")
    if value_metric not in {"kloc", "defects", "months"}:
        raise ValueError("value metric must be kloc, defects, or months")
    excluded = {"idX", "YearX", "Kloc+", "Effort-", "Defects-", "MonthsX"}
    if settings_columns is None:
        settings_columns = sorted({name for p in projects for name in p.attributes if name not in excluded})
    settings_columns = list(settings_columns)
    if not settings_columns or any(not name.strip() for name in settings_columns):
        raise ValueError("at least one setting column is required")
    missing = [name for name in settings_columns if any(name not in p.attributes for p in projects)]
    if missing:
        raise ValueError("unknown setting columns: %s" % ", ".join(missing))
    groups = {}
    for project in projects:
        key = tuple(project.attributes.get(name, "") for name in settings_columns)
        groups.setdefault(key, []).append(project)
    candidates = []
    for key, group in groups.items():
        if len(group) < minimum_group_size:
            continue
        effort = mean(p.effort for p in group)
        value = mean(getattr(p, value_metric) for p in group)
        candidates.append(SettingRecommendation(dict(zip(settings_columns, key)), len(group),
                                                mean(p.kloc for p in group), effort,
                                                value / effort if effort else 0.0,
                                                effort <= budget))
    if not candidates:
        raise ValueError("no setting combination meets the minimum group size")
    candidates.sort(key=lambda r: (-r.feasible, -r.value_per_effort, -r.mean_kloc,
                                   tuple(r.settings.values())))
    feasible = [r for r in candidates if r.feasible]
    if not feasible:
        raise ValueError("no setting combination fits the budget")
    return OptimizationResult(float(budget), value_metric, settings_columns,
                              candidates[:top], feasible[0])
