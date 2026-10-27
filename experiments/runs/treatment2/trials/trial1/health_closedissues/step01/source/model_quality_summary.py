#!/usr/bin/env python3
"""Summarize model-quality metrics and configuration settings from a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ColumnSummary:
    name: str
    kind: str
    count: int
    missing: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    values: dict[str, int] | None = None
    mode: str | None = None


def read_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read CSV rows while preserving source field names and values."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames), rows


def _number(value: str) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def summarize_column(name: str, values: Iterable[str]) -> ColumnSummary:
    raw = list(values)
    numeric = [_number(value) for value in raw]
    if all(value is not None for value in numeric):
        numbers = [value for value in numeric if value is not None]
        return ColumnSummary(name, "numeric", len(numbers), len(raw) - len(numbers),
                             min(numbers) if numbers else None,
                             max(numbers) if numbers else None,
                             statistics.fmean(numbers) if numbers else None,
                             statistics.median(numbers) if numbers else None)
    present = [value for value in raw if value and value.strip()]
    counts = Counter(present)
    return ColumnSummary(name, "categorical", len(present), len(raw) - len(present),
                         values=dict(sorted(counts.items())),
                         mode=counts.most_common(1)[0][0] if counts else None)


def classify_columns(fields: Sequence[str], metric_suffixes: tuple[str, ...] = ("+", "-")) -> tuple[list[str], list[str]]:
    """Classify quality metrics by suffix; remaining fields are configurations."""
    metrics = [field for field in fields if field.endswith(metric_suffixes)]
    return metrics, [field for field in fields if field not in metrics]


def summarize(path: str | Path) -> dict[str, Any]:
    fields, rows = read_rows(path)
    metric_fields, configuration_fields = classify_columns(fields)
    columns = [summarize_column(field, (row.get(field, "") for row in rows)) for field in fields]
    return {
        "source": str(path),
        "row_count": len(rows),
        "column_count": len(fields),
        "metric_fields": metric_fields,
        "configuration_fields": configuration_fields,
        "columns": [asdict(column) for column in columns],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, help="Write JSON summary to this path")
    args = parser.parse_args(argv)
    result = summarize(args.csv_path)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
