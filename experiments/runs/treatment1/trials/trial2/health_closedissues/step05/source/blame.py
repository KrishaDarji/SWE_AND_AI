#!/usr/bin/env python3
"""Describe hyperparameters associated with high MRE- in a CSV dataset."""
import argparse
import csv
import math
import os
import statistics
import sys


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"MRE-", "N_estimators", "criterion", "Min_sample_leaves",
                    "Min_impurity_decrease", "Max_depth"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("missing columns: " + ", ".join(sorted(missing)))
        rows = []
        for row in reader:
            try:
                row["MRE-"] = float(row["MRE-"])
                for name in ("N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"):
                    row[name] = float(row[name])
            except (TypeError, ValueError) as exc:
                raise ValueError("non-numeric value in row") from exc
            rows.append(row)
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def rank(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mid = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            result[order[k]] = mid
        i = j + 1
    return result


def correlation(x, y):
    if len(x) < 2:
        return 0.0
    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    return numerator / (denx * deny) if denx and deny else 0.0


def analyze(rows):
    errors = sorted(row["MRE-"] for row in rows)
    cutoff = errors[math.ceil(len(errors) * 0.75) - 1]
    high = [row for row in rows if row["MRE-"] >= cutoff]
    names = ["N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
    numeric = []
    for name in names:
        all_values = [r[name] for r in rows]
        high_values = [r[name] for r in high]
        numeric.append((name, statistics.mean(high_values), statistics.mean(all_values),
                        statistics.mean(high_values) - statistics.mean(all_values),
                        correlation(all_values, [r["MRE-"] for r in rows])))
    categories = {}
    for name in ("criterion",):
        for value in sorted({r[name] for r in rows}):
            group = [r for r in rows if r[name] == value]
            group_high = sum(r["MRE-"] >= cutoff for r in group)
            categories[(name, value)] = (len(group), group_high / len(group),
                                         group_high / len(high))
    return cutoff, len(high), numeric, categories


def format_report(path, rows):
    cutoff, high_count, numeric, categories = analyze(rows)
    lines = ["High-error hyperparameter association (descriptive)",
             "Dataset: " + path, "Rows: %d; high-error threshold (75th percentile): %.6f; high-error rows: %d" %
             (len(rows), cutoff, high_count), "", "Numeric parameters:",
             "parameter | high_mean | overall_mean | high_minus_overall | spearman_rho"]
    for name, hm, am, delta, rho in numeric:
        lines.append("%s | %.6f | %.6f | %+.6f | %+.4f" % (name, hm, am, delta, rho))
    lines += ["", "Categorical parameters:", "parameter=value | group_size | high-rate | share_of_high"]
    for (name, value), (size, rate, share) in categories.items():
        lines.append("%s=%s | %d | %.4f | %.4f" % (name, value, size, rate, share))
    lines += ["", "Interpretation: positive rho means larger numeric values tend to accompany higher MRE-.",
              "These are associations in observed configurations, not causal effects."]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    print(format_report(os.path.abspath(args.csv_path), read_rows(args.csv_path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
