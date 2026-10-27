"""Pure standard-library CSV loading and summary model."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class NumericSummary:
    count: int
    minimum: float
    maximum: float
    average: float
    median: float

    def as_dict(self) -> Dict[str, float]:
        return {"count": self.count, "minimum": self.minimum,
                "maximum": self.maximum, "average": self.average,
                "median": self.median}


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    setting_columns: List[str]
    measure_columns: List[str]
    measures: Dict[str, NumericSummary]
    setting_values: Dict[str, Dict[str, int]]

    def as_dict(self) -> dict:
        return {"rows": self.rows, "setting_columns": self.setting_columns,
                "measure_columns": self.measure_columns,
                "measures": {k: v.as_dict() for k, v in self.measures.items()},
                "setting_values": self.setting_values}


def _number(text: str, row_number: int, column: str) -> float:
    try:
        return float(text.strip())
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {column!r} is not numeric") from exc


def summarize_csv(path: str, measure_columns: Sequence[str] = ()) -> DatasetSummary:
    """Summarize a CSV; by default the last column is the measure."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header")
        headers = list(reader.fieldnames)
        selected = list(measure_columns) or [headers[-1]]
        unknown = [name for name in selected if name not in headers]
        if unknown:
            raise ValueError(f"unknown measure column(s): {', '.join(unknown)}")
        settings = [name for name in headers if name not in selected]
        values = {name: [] for name in selected}
        setting_values = {name: {} for name in settings}
        rows = 0
        for rows, record in enumerate(reader, 1):
            for name in selected:
                values[name].append(_number(record[name], rows + 1, name))
            for name in settings:
                value = record[name].strip()
                counts = setting_values[name]
                counts[value] = counts.get(value, 0) + 1
    measures = {name: NumericSummary(len(items), min(items), max(items),
                                      mean(items), median(items))
                for name, items in values.items()}
    return DatasetSummary(rows, settings, selected, measures, setting_values)


def render(summary: DatasetSummary) -> str:
    """Stable human-readable/JSON output for command-line reuse."""
    return json.dumps(summary.as_dict(), indent=2, sort_keys=True) + "\n"
