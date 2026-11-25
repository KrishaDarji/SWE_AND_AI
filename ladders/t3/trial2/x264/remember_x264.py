#!/usr/bin/env python3
"""Independent descriptive analysis for the x264 historical measurements."""
import argparse
import csv
import math
import os
import statistics
from typing import Dict, List, Tuple

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def load_dataset(path: str) -> Tuple[List[str], List[Dict[str, float]]]:
    if not os.path.isfile(path):
        raise ValueError("dataset does not exist: %s" % path)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(not name for name in reader.fieldnames):
            raise ValueError("CSV must have a non-empty header")
        fields = list(reader.fieldnames)
        rows = []
        for line_no, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError("line %d has more fields than the header" % line_no)
            if any(value is None or not value.strip() for value in raw.values()):
                raise ValueError("line %d contains a blank value" % line_no)
            row = {}
            for field, value in raw.items():
                try:
                    number = float(value.strip())
                except ValueError as exc:
                    raise ValueError("line %d field %s is not numeric" % (line_no, field)) from exc
                if not math.isfinite(number):
                    raise ValueError("line %d field %s is not finite" % (line_no, field))
                row[field] = number
            rows.append(row)
    if not rows:
        raise ValueError("CSV contains no data rows")
    return fields, rows


def identify_columns(fields: List[str]) -> Tuple[List[str], List[str]]:
    performance = [field for field in fields if "performance" in field.lower() or "throughput" in field.lower()]
    if not performance:
        raise ValueError("no performance column identified")
    if len(performance) != 1:
        raise ValueError("expected one performance column, found %s" % performance)
    config = [field for field in fields if field not in performance]
    if not config:
        raise ValueError("no configuration columns identified")
    return config, performance


def pearson(xs: List[float], ys: List[float]) -> float:
    sx, sy = statistics.mean(xs), statistics.mean(ys)
    dx = [x - sx for x in xs]
    dy = [y - sy for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return 0.0 if denom == 0 else sum(x * y for x, y in zip(dx, dy)) / denom


def analyze(path: str) -> str:
    fields, rows = load_dataset(path)
    config, performance = identify_columns(fields)
    target = performance[0]
    y = [row[target] for row in rows]
    lines = ["x264 remember analysis", "dataset=%s" % path, "rows=%d columns=%d" % (len(rows), len(fields))]
    lines.append("configuration_columns=%s" % ",".join(config))
    lines.append("performance_columns=%s" % ",".join(performance))
    lines.append("performance_summary=min=%.6f max=%.6f mean=%.6f median=%.6f stdev=%.6f" %
                 (min(y), max(y), statistics.mean(y), statistics.median(y), statistics.stdev(y)))
    binary_issues = []
    for field in config:
        values = [row[field] for row in rows]
        unique = sorted(set(values))
        if not all(value in (0.0, 1.0) for value in unique):
            binary_issues.append("%s=%s" % (field, unique))
        corr = pearson(values, y)
        lines.append("setting_summary %s min=%.6f max=%.6f mean=%.6f on=%d off=%d pearson=%.6f" %
                     (field, min(values), max(values), statistics.mean(values),
                      sum(value != 0 for value in values), sum(value == 0 for value in values), corr))
    # Constant settings have no estimable effect and can make least squares rank-deficient.
    model_config = [field for field in config if len({row[field] for row in rows}) > 1]
    x = [[row[field] for field in model_config] for row in rows]
    model = LinearRegression().fit(x, y)
    predictions = model.predict(x)
    ranked = sorted(zip(model_config, model.coef_), key=lambda pair: abs(pair[1]), reverse=True)
    lines.append("linear_regression train_r2=%.6f intercept=%.6f modeled_settings=%d omitted_constant_settings=%d" %
                 (r2_score(y, predictions), model.intercept_, len(model_config), len(config) - len(model_config)))
    lines.append("coefficient_ranking=" + ",".join("%s:%.6f" % pair for pair in ranked))
    lines.append("assumptions=rows are independent measurements; numeric settings are treated as quantitative indicators; higher performance is better")
    lines.append("limitations=correlations and in-sample regression do not establish causation or out-of-sample predictive accuracy")
    lines.append("data_quality=all values parsed as finite numbers; no missing values detected; binary_issues=%s" % ("none" if not binary_issues else ";".join(binary_issues)))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(analyze(args.dataset), end="")


if __name__ == "__main__":
    main()
