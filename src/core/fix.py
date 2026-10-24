"""Step 8: propose a small repair for a poor candidate."""

from __future__ import annotations

from statistics import mean, median
from typing import Dict, List, Sequence, Tuple

from .datasets import DatasetSpec
from .guess import predict_row
from .remember import as_number


def _score(spec: DatasetSpec, row: Dict[str, str], rows: Sequence[Dict[str, str]]) -> float:
    scores = []
    for objective in spec.objectives:
        values = [as_number(item[objective]) for item in rows if item.get(objective) not in (None, "", "?")]
        low, high = min(values), max(values)
        value = as_number(row[objective])
        normalized = 1.0 if high == low else (value - low) / (high - low)
        scores.append(normalized if objective.endswith("+") else 1.0 - normalized)
    return mean(scores)


def _replacement_values(rows: Sequence[Dict[str, str]], feature: str) -> List[str]:
    values = [row[feature] for row in rows if row.get(feature) not in (None, "", "?")]
    if not values:
        return []
    try:
        return [str(median(as_number(value) for value in values))]
    except ValueError:
        counts = {value: values.count(value) for value in set(values)}
        return [max(counts, key=counts.get)]


def fix(spec: DatasetSpec, rows: Sequence[Dict[str, str]], candidate: Dict[str, str]) -> Dict[str, object]:
    """Try one-feature changes and return the strongest improving repair."""

    baseline_score = _score(spec, candidate, rows)
    best_score = baseline_score
    best_row = dict(candidate)
    changed_feature = None
    features = [feature for feature in candidate if feature not in spec.objectives]
    for feature in features:
        for replacement in _replacement_values(rows, feature):
            if replacement == candidate.get(feature):
                continue
            trial = dict(candidate)
            trial[feature] = replacement
            predicted = predict_row(spec, rows, trial)
            trial.update({objective: str(value) for objective, value in predicted.items()})
            score = _score(spec, trial, rows)
            if score > best_score:
                best_score = score
                best_row = trial
                changed_feature = feature
    return {
        "dataset": spec.name,
        "baseline_score": baseline_score,
        "repaired_score": best_score,
        "improvement": best_score - baseline_score,
        "changed_feature": changed_feature,
        "repaired": changed_feature is not None,
        "candidate": best_row,
    }
