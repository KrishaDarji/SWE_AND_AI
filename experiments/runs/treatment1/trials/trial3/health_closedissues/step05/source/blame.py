#!/usr/bin/env python3
"""Describe hyperparameters associated with high model error."""
import argparse
import csv
import math
import statistics


NUMERIC = ("N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
ERROR = "MRE-"


def pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def load_rows(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or ERROR not in rows[0]:
        raise ValueError("CSV must contain an MRE- column and at least one row")
    return rows


def analyze(rows, high_fraction=0.20):
    errors = [float(row[ERROR]) for row in rows]
    cutoff = sorted(errors)[max(0, math.ceil(len(errors) * (1 - high_fraction)) - 1)]
    high = [row for row in rows if float(row[ERROR]) >= cutoff]
    result = {"rows": len(rows), "mean_error": statistics.mean(errors), "high_cutoff": cutoff,
              "high_count": len(high), "numeric": {}, "categorical": {}}
    for field in NUMERIC:
        values = [float(row[field]) for row in rows]
        high_values = [float(row[field]) for row in high]
        result["numeric"][field] = {"correlation": pearson(values, errors),
                                     "overall_mean": statistics.mean(values),
                                     "high_mean": statistics.mean(high_values)}
    categories = sorted({row["criterion"] for row in rows})
    for category in categories:
        subset = [row for row in rows if row["criterion"] == category]
        result["categorical"][category] = {"count": len(subset),
                                             "mean_error": statistics.mean(float(r[ERROR]) for r in subset)}
    return result


def render(result):
    lines = [f"Rows: {result['rows']}", f"Mean MRE-: {result['mean_error']:.6f}",
             f"High-error cutoff (top 20%): {result['high_cutoff']:.6f} ({result['high_count']} rows)",
             "Numeric hyperparameters (positive correlation/high mean indicates association with high error):"]
    for field, stats in result["numeric"].items():
        lines.append(f"- {field}: correlation={stats['correlation']:.4f}, overall_mean={stats['overall_mean']:.4f}, high_mean={stats['high_mean']:.4f}")
    lines.append("Categorical hyperparameters:")
    for field, stats in result["categorical"].items():
        lines.append(f"- criterion={field}: count={stats['count']}, mean_MRE-={stats['mean_error']:.6f}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(render(analyze(load_rows(args.dataset))), end="")


if __name__ == "__main__":
    main()
