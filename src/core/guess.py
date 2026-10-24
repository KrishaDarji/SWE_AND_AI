"""Step 2: predict objective values from nearby historical rows."""

from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Dict, List, Sequence

from .datasets import DatasetSpec
from .remember import as_number


def feature_columns(spec: DatasetSpec, rows: Sequence[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    return [column for column in rows[0] if column not in spec.objectives]


def _distance(left: Dict[str, str], right: Dict[str, str], columns: Sequence[str], ranges: Dict[str, float]) -> float:
    total = 0.0
    used = 0
    for column in columns:
        a, b = left.get(column, ""), right.get(column, "")
        if a in ("", "?") or b in ("", "?"):
            continue
        try:
            scale = ranges.get(column, 1.0) or 1.0
            difference = (as_number(a) - as_number(b)) / scale
        except ValueError:
            difference = 0.0 if a == b else 1.0
        total += difference * difference
        used += 1
    return sqrt(total / used) if used else float("inf")


def predict_row(
    spec: DatasetSpec,
    training_rows: Sequence[Dict[str, str]],
    target_row: Dict[str, str],
    k: int = 5,
) -> Dict[str, float]:
    """Predict every objective using the k nearest training rows."""

    columns = feature_columns(spec, training_rows)
    ranges: Dict[str, float] = {}
    for column in columns:
        values = []
        for row in training_rows:
            try:
                values.append(as_number(row[column]))
            except (KeyError, ValueError):
                pass
        ranges[column] = (max(values) - min(values)) if values else 1.0
    neighbors = sorted(
        ((_distance(target_row, row, columns, ranges), row) for row in training_rows),
        key=lambda item: item[0],
    )[: max(1, min(k, len(training_rows)))]
    predictions: Dict[str, float] = {}
    for objective in spec.objectives:
        values = [as_number(row[objective]) for _, row in neighbors if row.get(objective) not in (None, "", "?")]
        predictions[objective] = mean(values) if values else float("nan")
    return predictions


def leave_one_out_mae(spec: DatasetSpec, rows: Sequence[Dict[str, str]], k: int = 5) -> Dict[str, float]:
    """Evaluate the predictor without allowing a row to predict itself."""

    errors: Dict[str, List[float]] = {objective: [] for objective in spec.objectives}
    for index, target in enumerate(rows):
        training = list(rows[:index]) + list(rows[index + 1 :])
        prediction = predict_row(spec, training, target, k=k)
        for objective in spec.objectives:
            if target.get(objective) not in (None, "", "?"):
                errors[objective].append(abs(prediction[objective] - as_number(target[objective])))
    return {objective: mean(values) for objective, values in errors.items() if values}
