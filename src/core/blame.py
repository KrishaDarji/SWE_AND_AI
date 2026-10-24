"""Step 5: identify features associated with better or worse outcomes."""

from __future__ import annotations

from statistics import mean, median
from typing import Dict, List, Sequence

from .datasets import DatasetSpec
from .remember import as_number


def _oriented(value: float, objective: str) -> float:
    return value if objective.endswith("+") else -value


def _groups(rows: Sequence[Dict[str, str]], feature: str) -> List[List[Dict[str, str]]]:
    values = [row.get(feature, "") for row in rows if row.get(feature, "") not in ("", "?")]
    if not values:
        return []
    try:
        threshold = median(as_number(value) for value in values)
        low = [row for row in rows if row.get(feature, "") not in ("", "?") and as_number(row[feature]) <= threshold]
        high = [row for row in rows if row.get(feature, "") not in ("", "?") and as_number(row[feature]) > threshold]
        return [low, high]
    except ValueError:
        categories = sorted(set(values))
        return [[row for row in rows if row.get(feature) == category] for category in categories]


def blame(spec: DatasetSpec, rows: Sequence[Dict[str, str]], top_k: int = 10) -> List[Dict[str, object]]:
    """Rank features by the largest between-group outcome difference."""

    if not rows:
        return []
    features = [feature for feature in rows[0] if feature not in spec.objectives]
    findings: List[Dict[str, object]] = []
    for feature in features:
        groups = [group for group in _groups(rows, feature) if group]
        for objective in spec.objectives:
            group_means = []
            for group in groups:
                values = [as_number(row[objective]) for row in group if row.get(objective) not in (None, "", "?")]
                if values:
                    group_means.append((len(values), mean(values), group))
            if len(group_means) < 2:
                continue
            best = max(group_means, key=lambda item: _oriented(item[1], objective))
            worst = min(group_means, key=lambda item: _oriented(item[1], objective))
            effect = _oriented(best[1], objective) - _oriented(worst[1], objective)
            findings.append({
                "feature": feature,
                "objective": objective,
                "effect": effect,
                "best_group_mean": best[1],
                "worst_group_mean": worst[1],
                "best_group_size": best[0],
                "worst_group_size": worst[0],
            })
    return sorted(findings, key=lambda finding: float(finding["effect"]), reverse=True)[:top_k]
