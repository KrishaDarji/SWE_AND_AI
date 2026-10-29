"""Summarize x264 encoder settings and measured performance."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_rows(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = [{key: float(value.strip()) for key, value in row.items()} for row in reader]
    if not rows or not fields:
        raise ValueError("dataset is empty or has no header")
    return fields, rows


def summarize(path: Path) -> Dict:
    fields, rows = read_rows(path)
    target = "Performance-"
    features = [field for field in fields if field != target]
    x = [[row[field] for field in features] for row in rows]
    y = [row[target] for row in rows]
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(x, y)
    scaled = model[-1].coef_
    settings = {}
    for index, name in enumerate(features):
        values = [row[name] for row in rows]
        settings[name] = {"min": min(values), "max": max(values), "unique": len(set(values))}
    return {
        "rows": len(rows),
        "setting_columns": features,
        "quality_measure": {"name": target, "min": min(y), "max": max(y),
                             "mean": sum(y) / len(y)},
        "settings": settings,
        "standardized_linear_coefficients": dict(sorted(
            zip(features, scaled), key=lambda pair: abs(pair[1]), reverse=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    result = summarize(args.dataset)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"Historical runs: {result['rows']}")
    measure = result["quality_measure"]
    print(f"Quality measure: {measure['name']} (min={measure['min']:.3f}, "
          f"mean={measure['mean']:.3f}, max={measure['max']:.3f})")
    print("Encoder settings:")
    for name, stats in result["settings"].items():
        print(f"  {name}: min={stats['min']:.0f}, max={stats['max']:.0f}, "
              f"unique={stats['unique']}")
    print("Largest standardized linear associations:")
    for name, coefficient in list(result["standardized_linear_coefficients"].items())[:5]:
        print(f"  {name}: {coefficient:+.4f}")


if __name__ == "__main__":
    main()
