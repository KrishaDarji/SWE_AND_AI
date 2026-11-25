#!/usr/bin/env python3
"""Describe categorical project factors associated with unusually high effort."""
import argparse
import csv
import math
import statistics

ORDINAL = {"vl": 1, "l": 2, "n": 3, "h": 4, "vh": 5, "xh": 6}
EXCLUDE = {"idX", "Kloc+", "Effort-", "Defects-", "MonthsX"}

def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def numeric(value):
    try:
        return float(value.strip())
    except (AttributeError, ValueError):
        return None

def summarize(rows):
    efforts = [numeric(r["Effort-"]) for r in rows]
    threshold = statistics.quantiles(efforts, n=4, method="inclusive")[2]
    high = [e >= threshold for e in efforts]
    results = []
    for field in rows[0]:
        if field in EXCLUDE:
            continue
        groups = {}
        for row, effort, is_high in zip(rows, efforts, high):
            value = row[field].strip()
            groups.setdefault(value, []).append((effort, is_high))
        if len(groups) < 2:
            continue
        scored = []
        for value, items in groups.items():
            mean = statistics.mean(x[0] for x in items)
            rate = statistics.mean(x[1] for x in items)
            scored.append((value, len(items), mean, rate))
        scored.sort(key=lambda x: x[2], reverse=True)
        spread = scored[0][2] - scored[-1][2]
        results.append((spread, field, scored))
    return threshold, high.count(True), sorted(results, reverse=True)

def render(path):
    rows = load(path)
    if not rows:
        raise ValueError("dataset is empty")
    threshold, count, results = summarize(rows)
    lines = [f"Rows: {len(rows)}", f"Excessive-effort threshold (75th percentile): {threshold:.2f}",
             f"Projects at or above threshold: {count}", "", "Factors most associated with excessive effort:"]
    for spread, field, groups in results[:10]:
        high = max(groups, key=lambda x: x[2])
        low = min(groups, key=lambda x: x[2])
        lines.append(f"- {field}: highest {high[0]} (n={high[1]}, mean effort={high[2]:.2f}, high-rate={high[3]:.0%}); lowest {low[0]} (n={low[1]}, mean effort={low[2]:.2f}); spread={spread:.2f}")
    lines += ["", "Interpretation: these are associations from grouped observations, not causal effects. Numeric-looking fields are treated as categories when their values repeat."]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args()
    print(render(args.csv_path))

if __name__ == "__main__":
    main()
