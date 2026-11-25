#!/usr/bin/env python3
"""Independent historical configuration and quality summary for Health-ClosedIssues."""
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict

DATASET = "moot/optimize/hpo/Health-ClosedIssues0000.csv"


def load_dataset(path=DATASET):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields:
            raise ValueError("CSV has no header")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows")
    for number, row in enumerate(rows, 2):
        if set(row) != set(fields):
            raise ValueError("inconsistent columns at row %d" % number)
        if any(value is None or value.strip() == "" for value in row.values()):
            raise ValueError("blank value at row %d" % number)
    return fields, rows


def identify_columns(fields, rows):
    metrics = [field for field in fields if field.endswith(("+", "-"))]
    if not metrics:
        raise ValueError("no signed metric columns found")
    configs = [field for field in fields if field not in metrics]
    numeric_configs, categorical_configs = [], []
    for field in configs:
        try:
            [float(row[field]) for row in rows]
        except ValueError:
            categorical_configs.append(field)
        else:
            numeric_configs.append(field)
    targets = metrics[:]  # historical multi-objective quality targets
    return {"metrics": metrics, "targets": targets, "configs": configs,
            "numeric_configs": numeric_configs,
            "categorical_configs": categorical_configs}


def _numbers(rows, field):
    return [float(row[field]) for row in rows]


def _pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    dx = [x - mx for x in xs]; dy = [y - my for y in ys]
    den = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    return sum(x*y for x, y in zip(dx, dy)) / den if den else 0.0


def _rank(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1): ranks[order[k]] = rank
        i = j + 1
    return ranks


def summarize(path=DATASET):
    fields, rows = load_dataset(path)
    columns = identify_columns(fields, rows)
    summary = {"rows": len(rows), "columns": len(fields), "fields": fields,
               "columns_by_role": columns, "metrics": {}, "configs": {},
               "relationships": {}}
    for metric in columns["metrics"]:
        values = _numbers(rows, metric)
        summary["metrics"][metric] = {"min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values),
            "stdev": statistics.stdev(values), "best_direction": "lower" if metric.endswith("-") else "higher"}
    for config in columns["configs"]:
        values = [row[config] for row in rows]
        try:
            nums = [float(value) for value in values]
            summary["configs"][config] = {"type": "numeric", "min": min(nums), "median": statistics.median(nums),
                "mean": statistics.mean(nums), "max": max(nums), "unique": len(set(nums))}
            for metric in columns["metrics"]:
                summary["relationships"][config + "~" + metric] = {"pearson": _pearson(nums, _numbers(rows, metric)),
                    "spearman": _pearson(_rank(nums), _rank(_numbers(rows, metric)))}
        except ValueError:
            counts = Counter(values)
            summary["configs"][config] = {"type": "categorical", "unique": len(counts), "levels": dict(counts)}
            for metric in columns["metrics"]:
                grouped = defaultdict(list)
                for row in rows: grouped[row[config]].append(float(row[metric]))
                summary["relationships"][config + "~" + metric] = {"group_medians": {k: statistics.median(v) for k, v in sorted(grouped.items())}}
    return summary


def format_report(result):
    lines = ["Health-ClosedIssues historical remember summary", "rows=%d columns=%d" % (result["rows"], result["columns"]),
             "roles: metrics=%s; configs=%s; targets=%s" % (", ".join(result["columns_by_role"]["metrics"]), ", ".join(result["columns_by_role"]["configs"]), ", ".join(result["columns_by_role"]["targets"]))]
    for name, item in result["metrics"].items():
        lines.append("metric %s: direction=%s min=%.6g median=%.6g mean=%.6g max=%.6g stdev=%.6g" % (name, item["best_direction"], item["min"], item["median"], item["mean"], item["max"], item["stdev"]))
    for name, item in result["configs"].items():
        if item["type"] == "numeric": lines.append("config %s numeric: min=%.6g median=%.6g mean=%.6g max=%.6g unique=%d" % (name, item["min"], item["median"], item["mean"], item["max"], item["unique"]))
        else: lines.append("config %s categorical: unique=%d levels=%s" % (name, item["unique"], item["levels"]))
    for name, item in result["relationships"].items():
        if "pearson" in item: lines.append("relationship %s: pearson=%.6f spearman=%.6f" % (name, item["pearson"], item["spearman"]))
        else: lines.append("relationship %s: group_medians=%s" % (name, item["group_medians"]))
    lines += ["assumptions: signed column suffixes define metric direction; all other columns are configurations; all three metrics are retained as targets.",
              "limitations: associations are observational, not causal; duplicate or correlated trials may exist; no train/test generalization is claimed; categorical group sizes and missingness should be reviewed."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(format_report(summarize(sys.argv[1] if len(sys.argv) > 1 else DATASET)), end="")
