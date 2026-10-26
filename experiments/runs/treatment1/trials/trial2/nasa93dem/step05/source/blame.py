#!/usr/bin/env python3
"""Identify project factors associated with unusually high effort."""
import argparse
import csv
import math
import statistics
import sys


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no numeric values")
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            row = {key: value.strip() for key, value in row.items()}
            row["Effort-"] = float(row["Effort-"])
            rows.append(row)
    if not rows:
        raise ValueError("dataset has no rows")
    return rows


def association(rows, factor, threshold):
    groups = {}
    for row in rows:
        groups.setdefault(row[factor], []).append(row["Effort-"])
    overall = statistics.mean(row["Effort-"] for row in rows)
    records = []
    for value, efforts in groups.items():
        high = sum(e >= threshold for e in efforts)
        records.append({
            "value": value,
            "n": len(efforts),
            "median": statistics.median(efforts),
            "high_rate": high / len(efforts),
            "mean_delta": statistics.mean(efforts) - overall,
        })
    records.sort(key=lambda item: (item["high_rate"], item["median"]), reverse=True)
    # A weighted between-group variance is a compact effect-size-like ranking.
    between = sum(len(items) * (statistics.mean(items) - overall) ** 2
                  for items in groups.values())
    return between / len(rows), records


def analyze(path):
    rows = load_rows(path)
    effort = [row["Effort-"] for row in rows]
    threshold = percentile(effort, 0.75)
    target = "Effort-"
    factors = [name for name in rows[0] if name not in {"idX", target}]
    ranked = []
    for factor in factors:
        score, groups = association(rows, factor, threshold)
        ranked.append((score, factor, groups))
    ranked.sort(reverse=True, key=lambda item: item[0])
    return rows, threshold, ranked


def render(path):
    rows, threshold, ranked = analyze(path)
    high_count = sum(row["Effort-"] >= threshold for row in rows)
    lines = [
        "Excessive effort analysis (upper quartile rule)",
        f"Rows: {len(rows)}",
        f"Excessive-effort threshold (75th percentile): {threshold:.2f}",
        f"Projects at/above threshold: {high_count}",
        "",
        "Factors ranked by between-category effort separation:",
    ]
    for score, factor, groups in ranked:
        lines.append(f"- {factor}: score={score:.2f}")
        for group in groups[:3]:
            lines.append(
                f"  {group['value'] or '(blank)'}: n={group['n']}, "
                f"median_effort={group['median']:.2f}, "
                f"excessive_rate={group['high_rate']:.1%}"
            )
    lines += [
        "",
        "Interpretation: the leading factors are the strongest observed "
        "associations in this dataset, not causal effects."
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args(argv)
    try:
        print(render(args.dataset))
    except (OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
