"""Step 1: remember a MOOT dataset through summaries and rankings."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

from .datasets import DatasetSpec, read_rows


def as_number(value) -> float:
    return float(value)  # float() already ignores surrounding whitespace


def numeric_values(rows: Sequence[Dict[str, str]], column: str) -> List[float]:
    return [as_number(row[column]) for row in rows if row.get(column, "") not in ("", "?")]


def summarize(
    spec: DatasetSpec, limit: Optional[int] = None, rows: Optional[Sequence[Dict[str, str]]] = None
) -> Dict[str, object]:
    """Return reproducible descriptive statistics for the dataset (or the given rows)."""

    rows = read_rows(spec, limit=limit) if rows is None else rows
    objectives = {}
    for objective in spec.objectives:
        values = numeric_values(rows, objective)
        if values:
            objectives[objective] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": mean(values),
                "std": pstdev(values),
            }
    return {"dataset": spec.name, "rows": len(rows), "objectives": objectives}


def bounds(spec: DatasetSpec, rows: Sequence[Dict[str, str]]) -> Dict[str, Tuple[float, float]]:
    """Observed (min, max) of every objective, used to normalize scores."""

    limits = {}
    for objective in spec.objectives:
        values = numeric_values(rows, objective)
        if values:
            limits[objective] = (min(values), max(values))
    return limits


def score(spec: DatasetSpec, row: Dict[str, object], limits: Dict[str, Tuple[float, float]]) -> float:
    """Mean normalized objective score; 1 is best, 0 is worst."""

    scores = []
    for objective, (low, high) in limits.items():
        if row.get(objective, "") in ("", "?"):
            continue
        normalized = 1.0 if high == low else (as_number(row[objective]) - low) / (high - low)
        scores.append(1.0 - normalized if objective.endswith("-") else normalized)
    return mean(scores) if scores else float("nan")


def rank_rows(
    spec: DatasetSpec, rows: Sequence[Dict[str, str]], top_k: int = 5
) -> List[Tuple[float, Dict[str, str]]]:
    """Rank rows by normalized objective score; higher score is better."""

    limits = bounds(spec, rows)
    scored = [(score(spec, row, limits), row) for row in rows]
    scored = [item for item in scored if item[0] == item[0]]  # drop NaN (no objectives)
    return sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
