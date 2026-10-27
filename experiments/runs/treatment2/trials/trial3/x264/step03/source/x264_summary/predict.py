"""Nearest-neighbor prediction for a new x264 configuration."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Prediction:
    value: float
    neighbors: int
    distance: float


def _number(text: str, row: int, column: str) -> float:
    try:
        return float(text.strip())
    except ValueError as exc:
        raise ValueError(f"row {row}: {column!r} is not numeric") from exc


def predict_encoding_time(path: str, configuration: Mapping[str, object],
                          measure: str = "", neighbors: int = 5) -> Prediction:
    """Predict a measure using the mean of closest normalized numeric settings.

    Binary/categorical settings use exact-match distance; numeric settings use
    range normalization. Ties are deterministic in source-row order.
    """
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header")
        headers = list(reader.fieldnames)
        target = measure or headers[-1]
        if target not in headers:
            raise ValueError(f"unknown measure column: {target}")
        settings = [name for name in headers if name != target]
        missing = [name for name in settings if name not in configuration]
        if missing:
            raise ValueError(f"missing setting(s): {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV must contain at least one data row")
    numeric = {name: [_number(row[name], i + 2, name) for i, row in enumerate(rows)]
               for name in settings}
    spans = {name: max(values) - min(values) for name, values in numeric.items()}
    target_values = {name: _number(str(configuration[name]), 0, name)
                     for name in settings}
    scored = []
    for index, row in enumerate(rows):
        distance = 0.0
        for name in settings:
            span = spans[name] or 1.0
            distance += abs(numeric[name][index] - target_values[name]) / span
        scored.append((distance, index, _number(row[target], index + 2, target)))
    scored.sort(key=lambda item: (item[0], item[1]))
    chosen = scored[:min(neighbors, len(scored))]
    return Prediction(sum(item[2] for item in chosen) / len(chosen),
                      len(chosen), chosen[0][0])
