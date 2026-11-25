#!/usr/bin/env python3
"""Choose the fastest quality-preserving configuration in the x264 table."""

import csv
import json
import math
import sys
from pathlib import Path


DATASET = Path(__file__).resolve().parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"
OBJECTIVE = "Performance-"
QUALITY_FLAGS = (
    "no_8x8dct", "no_cabac", "no_deblock", "no_mbtree",
    "no_mixed_refs", "no_weightb",
)
REQUIRED_COLUMNS = (
    "no_asm", "no_8x8dct", "no_cabac", "no_deblock", "no_fast_pskip",
    "no_mbtree", "no_mixed_refs", "no_weightb", "rc_lookahead",
    "rc_lookahead_20", "rc_lookahead_40", "rc_lookahead_60", "ref",
    "ref_1", "ref_5", "ref_9", OBJECTIVE,
)


def load_dataset(path=DATASET):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
            raise ValueError("unexpected or missing CSV columns")
        rows = list(reader)
    if not rows:
        raise ValueError("dataset is empty")
    for number, row in enumerate(rows, 2):
        for column in REQUIRED_COLUMNS[:-1]:
            if row[column].strip() not in {"0", "1"}:
                raise ValueError(f"row {number}: {column} must be 0 or 1")
        try:
            value = float(row[OBJECTIVE].strip())
        except ValueError as exc:
            raise ValueError(f"row {number}: invalid {OBJECTIVE}") from exc
        if not math.isfinite(value):
            raise ValueError(f"row {number}: non-finite {OBJECTIVE}")
    return rows


def is_feasible(row):
    # Quality constraint: do not disable any of the quality-affecting tools.
    return all(row[name].strip() == "0" for name in QUALITY_FLAGS)


def choose(rows):
    feasible = [row for row in rows if is_feasible(row)]
    if not feasible:
        return None, 0
    selected = min(feasible, key=lambda row: (float(row[OBJECTIVE].strip()), tuple(row[c].strip() for c in REQUIRED_COLUMNS[:-1])))
    return selected, len(feasible)


def main(path=DATASET):
    rows = load_dataset(path)
    selected, feasible_count = choose(rows)
    result = {
        "dataset": str(path), "rows": len(rows), "feasible_rows": feasible_count,
        "quality_constraint": "no_8x8dct=no_cabac=no_deblock=no_mbtree=no_mixed_refs=no_weightb=0",
        "objective": "minimize Performance-",
    }
    if selected is None:
        result["selected"] = None
        result["message"] = "No configuration satisfies the quality constraint."
    else:
        result["selected"] = {c: selected[c].strip() for c in REQUIRED_COLUMNS[:-1]}
        result["quality_value"] = "all six quality-affecting disable flags are 0"
        result["performance_value"] = float(selected[OBJECTIVE].strip())
    print(json.dumps(result, sort_keys=True))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET))
