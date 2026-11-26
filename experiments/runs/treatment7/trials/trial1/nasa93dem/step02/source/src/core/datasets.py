"""Dataset loading and schema inspection shared by every domain."""

from __future__ import annotations

import csv
import random
from typing import Dict, List, Optional, Sequence

from .schema import ROOT, DatasetSpec  # noqa: F401  (re-exported)
from src.domains import DATASETS


def get_dataset(name: str) -> DatasetSpec:
    """Return a registered dataset or raise a useful error."""

    try:
        return DATASETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {name!r}; choose from: {choices}") from exc


def read_rows(spec: DatasetSpec, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Read CSV rows (header and values stripped) without changing source values."""

    if not spec.path.exists():
        raise FileNotFoundError(f"Dataset not found: {spec.path}")
    with spec.path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no header: {spec.path}")
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({key: value.strip() for key, value in row.items()})
            if limit is not None and len(rows) >= limit:
                break
        return rows


def sample_rows(spec: DatasetSpec, n: Optional[int] = None, seed: int = 1) -> List[Dict[str, str]]:
    """A reproducible random sample in random order (files are often sorted)."""

    rows = read_rows(spec)
    n = min(n or spec.max_rows, len(rows))
    return random.Random(seed).sample(rows, n)


def features(spec: DatasetSpec, rows: Sequence[Dict[str, str]]) -> List[str]:
    """Input columns: not an objective, and not ignored (MOOT names ending in X)."""

    if not rows:
        return []
    return [column for column in rows[0] if column not in spec.objectives and not column.endswith("X")]


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
