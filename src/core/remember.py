"""Step 1: remember a MOOT dataset through summaries and rankings."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

from .datasets import DatasetSpec, read_rows


def as_number(value: str) -> float:
    return float(value.strip())


def numeric_values(rows: Sequence[Dict[str, str]], column: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(column, "")
        if value not in ("", "?"):
            values.append(as_number(value))
    return values


def summarize(spec: DatasetSpec, limit: Optional[int] = None) -> Dict[str, object]:
    """Return reproducible descriptive statistics for the dataset."""

    rows = read_rows(spec, limit=limit)
    result: Dict[str, object] = {"dataset": spec.name, "rows": len(rows), "objectives": {}}
    objectives = result["objectives"]
    assert isinstance(objectives, dict)
    for objective in spec.objectives:
        values = numeric_values(rows, objective)
        if not values:
            continue
        objectives[objective] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
            "std": pstdev(values),
        }
    return result


def rank_rows(
    spec: DatasetSpec, rows: Sequence[Dict[str, str]], top_k: int = 5
) -> List[Tuple[float, Dict[str, str]]]:
    """Rank rows by normalized objective score; higher score is better."""

    columns = spec.objectives
    values = {column: numeric_values(rows, column) for column in columns}
    bounds = {column: (min(v), max(v)) for column, v in values.items() if v}

    scored = []
    for row in rows:
        scores = []
        for column in columns:
            if column not in bounds or row.get(column, "") in ("", "?"):
                continue
            low, high = bounds[column]
            value = as_number(row[column])
            normalized = 1.0 if high == low else (value - low) / (high - low)
            if column.endswith("-"):
                normalized = 1.0 - normalized
            scores.append(normalized)
        if scores:
            scored.append((mean(scores), row))
    return sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
