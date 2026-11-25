#!/usr/bin/env python3
"""Reusable, standard-library-only summaries for experiment CSV datasets."""

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class ColumnSummary:
    name: str
    kind: str
    count: int
    unique: int
    minimum: float = None
    maximum: float = None
    mean: float = None
    median: float = None
    values: Dict[str, int] = None


def _number(value: str):
    try:
        number = float(value.strip())
        return number if math.isfinite(number) else None
    except (AttributeError, ValueError):
        return None


def summarize_csv(path: Path) -> dict:
    """Return a JSON-serializable summary; this is the reusable API boundary."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        rows = list(reader)
    summaries = []
    for name in reader.fieldnames:
        raw = [(row.get(name) or "").strip() for row in rows]
        numbers = [_number(value) for value in raw]
        numeric = all(value is not None for value in numbers)
        distinct = set(raw)
        binary = numeric and distinct <= {"0", "1"}
        if binary:
            kind = "encoder_setting"
        elif numeric:
            kind = "quality_measure"
        else:
            kind = "categorical"
        summary = ColumnSummary(name=name, kind=kind, count=len(raw),
                                unique=len(distinct))
        if numeric:
            summary.minimum = min(numbers)
            summary.maximum = max(numbers)
            summary.mean = statistics.fmean(numbers)
            summary.median = statistics.median(numbers)
        elif len(distinct) <= 20:
            summary.values = dict(sorted((item, raw.count(item)) for item in distinct))
        summaries.append(summary)
    return {
        "dataset": str(path),
        "rows": len(rows),
        "columns": len(reader.fieldnames),
        "encoder_settings": [asdict(item) for item in summaries if item.kind == "encoder_setting"],
        "quality_measures": [asdict(item) for item in summaries if item.kind == "quality_measure"],
        "other_columns": [asdict(item) for item in summaries if item.kind == "categorical"],
    }


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = summarize_csv(args.csv_path)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
