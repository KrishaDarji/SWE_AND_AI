"""Robust detection of unusually high or low benchmark measurements."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import List


@dataclass(frozen=True)
class Anomaly:
    row_number: int
    configuration: dict
    measure: str
    value: float
    score: float
    reason: str


def detect_anomalies(path: str, measure: str = "", threshold: float = 3.5) -> List[Anomaly]:
    """Return rows whose modified z-score exceeds ``threshold``.

    The score is based on median absolute deviation (MAD), making it less
    sensitive to the very outliers it is intended to find. If MAD is zero,
    a positive IQR-based scale is used; a constant dataset has no anomalies.
    """
    if threshold <= 0 or not math.isfinite(threshold):
        raise ValueError("threshold must be a positive finite number")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header")
        headers = list(reader.fieldnames)
        target = measure or headers[-1]
        if target not in headers:
            raise ValueError(f"unknown measure column: {target}")
        settings = [name for name in headers if name != target]
        rows = []
        for row_number, row in enumerate(reader, 2):
            try:
                value = float(str(row[target]).strip())
            except ValueError as exc:
                raise ValueError(f"row {row_number}: {target!r} is not numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"row {row_number}: {target!r} must be finite")
            rows.append((row_number, row, value))
    if not rows:
        return []
    values = [item[2] for item in rows]
    center = median(values)
    deviations = [abs(value - center) for value in values]
    mad = median(deviations)
    if mad:
        scale = mad / 0.6744897501960817
        score = lambda value: abs(value - center) / scale
    else:
        ordered = sorted(values)
        q1 = median(ordered[:len(ordered) // 2])
        q3 = median(ordered[(len(ordered) + 1) // 2:])
        iqr = q3 - q1
        if not iqr:
            return []
        score = lambda value: abs(value - center) / (iqr / 1.349)
    result = []
    for row_number, row, value in rows:
        current = score(value)
        if current > threshold:
            direction = "high" if value > center else "low"
            result.append(Anomaly(row_number, {name: row[name].strip() for name in settings},
                                  target, value, current, f"{direction} robust outlier"))
    return result
