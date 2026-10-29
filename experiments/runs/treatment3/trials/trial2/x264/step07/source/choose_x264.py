#!/usr/bin/env python3
"""Choose the fastest measured x264 configuration satisfying a quality policy."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

QUALITY_POLICY = {
    "no_cabac": 0,
    "no_deblock": 0,
    "no_mbtree": 0,
    "no_mixed_refs": 0,
    "no_weightb": 0,
    "ref_9": 0,
}
PERFORMANCE_COLUMN = "Performance-"


def load_dataset(path: Path) -> Tuple[List[Dict[str, int]], List[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or PERFORMANCE_COLUMN not in reader.fieldnames:
            raise ValueError("dataset must contain a Performance- column")
        setting_names = [name for name in reader.fieldnames if name != PERFORMANCE_COLUMN]
        if not setting_names:
            raise ValueError("dataset has no setting columns")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            try:
                row = {name: int(raw[name].strip()) for name in setting_names}
                performance = float(raw[PERFORMANCE_COLUMN].strip())
            except (AttributeError, KeyError, ValueError) as exc:
                raise ValueError(f"invalid row {line_number}: {exc}") from exc
            if any(value not in (0, 1) for value in row.values()) or performance != performance:
                raise ValueError(f"invalid binary setting or performance at row {line_number}")
            row[PERFORMANCE_COLUMN] = performance
            rows.append(row)
    if not rows:
        raise ValueError("dataset is empty")
    missing = set(QUALITY_POLICY) - set(setting_names)
    if missing:
        raise ValueError(f"quality columns missing: {sorted(missing)}")
    return rows, setting_names


def choose(rows: List[Dict[str, int]]) -> Dict[str, int]:
    feasible = [row for row in rows if all(row[key] == value for key, value in QUALITY_POLICY.items())]
    if not feasible:
        raise LookupError("no configuration satisfies the quality constraint")
    return min(feasible, key=lambda row: row[PERFORMANCE_COLUMN])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    rows, settings = load_dataset(args.dataset)
    selected = choose(rows)
    print("quality_constraint=preserve_quality_features_at_zero")
    print("quality_policy=" + ",".join(f"{k}={QUALITY_POLICY[k]}" for k in QUALITY_POLICY))
    print("selected_settings=" + ",".join(f"{k}={selected[k]}" for k in settings))
    print(f"quality_value={sum(selected[k] == v for k, v in QUALITY_POLICY.items())}/{len(QUALITY_POLICY)}")
    print(f"performance_value={selected[PERFORMANCE_COLUMN]:.3f}")
    print(f"feasible_count={sum(all(row[k] == v for k, v in QUALITY_POLICY.items()) for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
