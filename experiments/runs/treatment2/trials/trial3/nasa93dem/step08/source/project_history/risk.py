"""Recommend a lower-risk process change at the lowest observed cost."""

from statistics import median
from typing import Iterable

from .model import Project


def recommend_cheapest_risk_change(projects: Iterable[Project], attribute: str,
                                   current_value: str) -> dict[str, object]:
    """Find the least-cost process group with lower median defects/KLOC.

    Risk is the median observed defects per KLOC and cost is median effort per
    KLOC.  Both measures require positive KLOC.  This is an observational
    recommendation, not a causal claim; ties prefer the first value in sorted
    order.  A missing or unsupported current process returns no recommendation.
    """
    groups: dict[str, list[tuple[float, float]]] = {}
    for item in projects:
        value = item.attributes.get(attribute, "").strip()
        if (not value or item.kloc is None or item.kloc <= 0 or
                item.defects is None or item.effort is None):
            continue
        groups.setdefault(value, []).append((item.defects / item.kloc,
                                              item.effort / item.kloc))
    if current_value not in groups:
        return {"attribute": attribute, "current_value": current_value,
                "current_risk": None, "current_cost": None,
                "recommendation": None, "reason": "insufficient data"}
    current = groups[current_value]
    current_risk, current_cost = median(x[0] for x in current), median(x[1] for x in current)
    alternatives = []
    for value, observations in groups.items():
        if value == current_value:
            continue
        risk = median(x[0] for x in observations)
        cost = median(x[1] for x in observations)
        if risk < current_risk:
            alternatives.append((cost, value, risk))
    if not alternatives:
        recommendation = None
        reason = "no observed lower-risk alternative"
    else:
        cost, value, risk = min(alternatives, key=lambda item: (item[0], item[1]))
        recommendation = {"value": value, "risk": risk, "cost": cost,
                          "risk_reduction": current_risk - risk,
                          "cost_change": cost - current_cost}
        reason = "lowest-cost observed alternative with lower risk"
    return {"attribute": attribute, "current_value": current_value,
            "current_risk": current_risk, "current_cost": current_cost,
            "recommendation": recommendation, "reason": reason}
