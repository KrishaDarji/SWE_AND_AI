"""Step 8: propose a small repair for a poor candidate."""

from __future__ import annotations

from statistics import median
from typing import Dict, List, Sequence

from .datasets import DatasetSpec, features
from .guess import predict_row
from .remember import as_number, bounds, score


def _replacement_values(rows: Sequence[Dict[str, str]], feature: str) -> List[str]:
    """The typical value of a feature: median if numeric, else the most common value."""
    values = [row[feature] for row in rows if row.get(feature) not in (None, "", "?")]
    if not values:
        return []
    try:
        return [format(median(as_number(value) for value in values), "g")]  # 0.0 -> "0"
    except ValueError:
        return [max(set(values), key=values.count)]


def fix(spec: DatasetSpec, rows: Sequence[Dict[str, str]], candidate: Dict[str, str]) -> Dict[str, object]:
    """Try one-feature changes (towards typical values) and keep the best predicted repair."""

    limits = bounds(spec, rows)
    baseline_score = score(spec, candidate, limits)
    best_score, best_row, changed_feature = baseline_score, dict(candidate), None
    for feature in features(spec, rows):
        for replacement in _replacement_values(rows, feature):
            if replacement == candidate.get(feature):
                continue
            trial = dict(candidate)
            trial[feature] = replacement
            trial.update({objective: str(value) for objective, value in predict_row(spec, rows, trial).items()})
            trial_score = score(spec, trial, limits)
            if trial_score > best_score:
                best_score, best_row, changed_feature = trial_score, trial, feature
    return {
        "dataset": spec.name,
        "baseline_score": baseline_score,
        "repaired_score": best_score,
        "improvement": best_score - baseline_score,
        "changed_feature": changed_feature,
        "old_value": candidate.get(changed_feature) if changed_feature else None,
        "new_value": best_row.get(changed_feature) if changed_feature else None,
        "repaired": changed_feature is not None,
        "candidate": best_row,
    }
