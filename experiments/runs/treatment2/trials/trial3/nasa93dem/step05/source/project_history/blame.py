"""Descriptive evidence about factors associated with excessive effort."""

from statistics import median
from typing import Iterable, Optional

from .model import Project


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + end - 1) / 2.0 + 1
        for position in range(index, end):
            ranks[ordered[position][0]] = average
        index = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> Optional[float]:
    if len(left) < 2:
        return None
    left_rank, right_rank = _rank(left), _rank(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_rank, right_rank))
    left_scale = sum((a - left_mean) ** 2 for a in left_rank)
    right_scale = sum((b - right_mean) ** 2 for b in right_rank)
    return numerator / (left_scale * right_scale) ** 0.5 if left_scale and right_scale else None


def explain_excessive_effort(projects: Iterable[Project], *, threshold: Optional[float] = None) -> dict[str, object]:
    """Rank observed associations with effort above a robust dataset threshold.

    Categorical attributes are compared by excess-rate difference between the
    value with the highest rate and the value with the lowest rate. Numeric
    fields are ranked by Spearman correlation with effort. This is descriptive,
    not causal, and excludes records missing effort.
    """
    items = [project for project in projects if project.effort is not None]
    if not items:
        return {"threshold": threshold, "excessive_count": 0, "categorical": [], "numeric": []}
    cutoff = median(item.effort for item in items) if threshold is None else float(threshold)
    excessive = [item for item in items if item.effort > cutoff]
    names = sorted({name for item in items for name in item.attributes})
    categorical = []
    for name in names:
        groups = {}
        for item in items:
            value = item.attributes.get(name, "").strip()
            if value:
                groups.setdefault(value, []).append(item.effort > cutoff)
        if len(groups) < 2:
            continue
        rates = {value: sum(flags) / len(flags) for value, flags in groups.items()}
        high, low = max(rates, key=rates.get), min(rates, key=rates.get)
        categorical.append({"factor": name, "high_value": high, "high_rate": rates[high],
                            "low_value": low, "low_rate": rates[low],
                            "rate_difference": rates[high] - rates[low], "sample_size": len(items)})
    categorical.sort(key=lambda result: (-result["rate_difference"], result["factor"]))
    numeric = []
    numeric_fields = {"YearX": lambda item: item.year, "Kloc+": lambda item: item.kloc,
                      "Defects-": lambda item: item.defects, "MonthsX": lambda item: item.months}
    for name, getter in numeric_fields.items():
        paired = [(getter(item), item.effort) for item in items if getter(item) is not None]
        correlation = _correlation([pair[0] for pair in paired], [pair[1] for pair in paired])
        if correlation is not None:
            numeric.append({"factor": name, "spearman": correlation, "sample_size": len(paired)})
    numeric.sort(key=lambda result: (-abs(result["spearman"]), result["factor"]))
    return {"threshold": cutoff, "excessive_count": len(excessive),
            "categorical": categorical, "numeric": numeric}
