"""Reusable CSV summary and pre-evaluation quality prediction utilities."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _summarize_numeric(values):
    ordered = sorted(values)
    return {"count": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "minimum": ordered[0],
            "maximum": ordered[-1]}


def summarize_rows(rows, quality_columns=None, configuration_columns=None):
    rows = list(rows)
    if not rows:
        raise ValueError("CSV contains no data rows")
    columns = list(rows[0])
    quality_columns = list(quality_columns or columns[:3])
    configuration_columns = list(configuration_columns or columns[3:])
    missing = [c for c in quality_columns + configuration_columns if c not in columns]
    if missing:
        raise ValueError("Unknown columns: " + ", ".join(missing))
    quality = {}
    for column in quality_columns:
        usable = [_number(row[column]) for row in rows]
        usable = [v for v in usable if v is not None]
        quality[column] = _summarize_numeric(usable) if usable else {"count": 0}
    configuration = {}
    for column in configuration_columns:
        values = [row[column] for row in rows]
        numeric = [_number(v) for v in values]
        if all(v is not None for v in numeric):
            configuration[column] = {"kind": "numeric", **_summarize_numeric(numeric)}
        else:
            counts = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            configuration[column] = {"kind": "categorical", "count": len(values),
                                     "unique": len(counts),
                                     "values": dict(sorted(counts.items()))}
    return {"row_count": len(rows), "quality_metrics": quality,
            "configuration": configuration}


def summarize_csv(path, quality_columns=None, configuration_columns=None):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        return summarize_rows(reader, quality_columns, configuration_columns)


def predict_quality(rows, target_configuration, quality_columns=("MRE-", "ACC+"),
                    configuration_columns=None, neighbors=25):
    """Predict quality from similar completed configurations before evaluation."""
    rows = list(rows)
    if not rows:
        raise ValueError("CSV contains no data rows")
    configuration_columns = list(configuration_columns or
                                 [c for c in rows[0] if c not in quality_columns])
    missing = [c for c in list(quality_columns) + configuration_columns
               if c not in rows[0]]
    missing += [c for c in configuration_columns if c not in target_configuration]
    if missing:
        raise ValueError("Unknown columns: " + ", ".join(dict.fromkeys(missing)))

    numeric_ranges = {}
    for column in configuration_columns:
        values = [_number(row[column]) for row in rows]
        if all(value is not None for value in values):
            span = max(values) - min(values)
            numeric_ranges[column] = (min(values), span or 1.0)

    scored = []
    for row in rows:
        distance = 0.0
        for column in configuration_columns:
            if column in numeric_ranges:
                low, span = numeric_ranges[column]
                distance += ((_number(row[column]) - _number(target_configuration[column])) / span) ** 2
            elif row[column] != str(target_configuration[column]):
                distance += 1.0
        quality = [_number(row[column]) for column in quality_columns]
        if all(value is not None for value in quality):
            scored.append((math.sqrt(distance), quality))
    if not scored:
        raise ValueError("No rows have usable quality metrics")
    scored.sort(key=lambda item: item[0])
    chosen = scored[:max(1, min(neighbors, len(scored)))]
    weights = [1.0 / (distance + 1e-9) for distance, _ in chosen]
    prediction = [sum(weight * quality[index] for weight, (_, quality) in zip(weights, chosen)) /
                  sum(weights) for index in range(len(quality_columns))]
    return {"predictions": dict(zip(quality_columns, prediction)),
            "neighbors_used": len(chosen), "nearest_distance": chosen[0][0],
            "baseline": {column: statistics.fmean(item[1][index] for item in scored)
                         for index, column in enumerate(quality_columns)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--predict", nargs="*", metavar="KEY=VALUE")
    parser.add_argument("--quality", nargs="+")
    parser.add_argument("--configuration", nargs="+")
    args = parser.parse_args(argv)
    quality = args.quality or ["MRE-", "ACC+"]
    configuration = args.configuration or ["N_estimators", "criterion", "Min_sample_leaves",
                                            "Min_impurity_decrease", "Max_depth"]
    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.predict is not None:
        target = dict(item.split("=", 1) for item in args.predict)
        result = predict_quality(rows, target, quality, configuration)
    else:
        result = summarize_rows(rows, quality, configuration)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
