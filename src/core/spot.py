"""Step 9: detect unusual records and possible data-quality problems."""

from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Dict, List, Sequence

from .datasets import DatasetSpec
from .remember import as_number


def _numeric_columns(rows: Sequence[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    columns = []
    for column in rows[0]:
        values = [row.get(column, "") for row in rows if row.get(column, "") not in ("", "?")]
        try:
            for value in values:
                as_number(value)
            if values:
                columns.append(column)
        except ValueError:
            pass
    return columns


def spot(spec: DatasetSpec, rows: Sequence[Dict[str, str]], top_k: int = 10) -> List[Dict[str, object]]:
    """Rank rows by their combined absolute z-score across numeric columns."""

    columns = _numeric_columns(rows)
    stats = {}
    for column in columns:
        values = [as_number(row[column]) for row in rows if row.get(column) not in (None, "", "?")]
        stats[column] = (mean(values), pstdev(values))
    findings = []
    for index, row in enumerate(rows):
        deviations = []
        for column in columns:
            value = row.get(column, "")
            if value in ("", "?"):
                continue
            center, spread = stats[column]
            if spread > 0:
                deviations.append((as_number(value) - center) / spread)
        if deviations:
            score = sqrt(mean(deviation * deviation for deviation in deviations))
            findings.append({
                "row_index": index,
                "score": score,
                "largest_deviation": max(deviations, key=abs),
                "row": row,
            })
    return sorted(findings, key=lambda finding: float(finding["score"]), reverse=True)[:top_k]
