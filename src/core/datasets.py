"""Dataset registry and schema inspection for the selected MOOT tasks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    client: str
    path: Path
    objectives: List[str]


DATASETS: Dict[str, DatasetSpec] = {
    "x264": DatasetSpec(
        name="x264",
        client="video-platform engineer",
        path=ROOT / "moot/optimize/config/X264_AllMeasurements.csv",
        objectives=["Performance-"],
    ),
    "nasa93dem": DatasetSpec(
        name="nasa93dem",
        client="software project planning manager",
        path=ROOT / "moot/optimize/process/nasa93dem.csv",
        objectives=["Effort-", "Defects-"],
    ),
    "health_closedissues": DatasetSpec(
        name="health_closedissues",
        client="open-source project maintainer",
        path=ROOT / "moot/optimize/hpo/Health-ClosedIssues0000.csv",
        objectives=["MRE-", "ACC+", "PRED40+"],
    ),
}


def get_dataset(name: str) -> DatasetSpec:
    """Return a registered dataset or raise a useful error."""

    try:
        return DATASETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {name!r}; choose from: {choices}") from exc


def _clean_header(value: str) -> str:
    return value.strip()


def read_rows(spec: DatasetSpec, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Read CSV rows without changing source values."""

    if not spec.path.exists():
        raise FileNotFoundError(f"Dataset not found: {spec.path}")
    with spec.path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no header: {spec.path}")
        reader.fieldnames = [_clean_header(field) for field in reader.fieldnames]
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({_clean_header(k): v.strip() for k, v in row.items()})
            if limit is not None and len(rows) >= limit:
                break
        return rows


def schema(spec: DatasetSpec) -> Dict[str, object]:
    """Return a compact, JSON-friendly schema summary."""

    rows = read_rows(spec, limit=100)
    columns = list(rows[0]) if rows else []
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    for column in columns:
        values = [row[column] for row in rows if row[column] not in ("", "?")]
        try:
            for value in values:
                float(value)
            numeric_columns.append(column)
        except ValueError:
            categorical_columns.append(column)
    return {
        "name": spec.name,
        "path": str(spec.path),
        "rows_sampled": len(rows),
        "columns": columns,
        "objectives": spec.objectives,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }
