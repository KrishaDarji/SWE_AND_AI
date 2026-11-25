#!/usr/bin/env python3
"""Explain which nasa93dem project characteristics are associated with high effort."""
import argparse
import csv
import math
from pathlib import Path

TARGET = "Effort-"
OUTPUTS = {"idX", "Kloc+", "Effort-", "Defects-", "MonthsX"}
ORDER = {"vl": 1, "l": 2, "n": 3, "h": 4, "vh": 5, "xh": 6}


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            result[order[k]] = rank
        i = j + 1
    return result


def correlation(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return numerator / (da * db) if da and db else 0.0


def numeric(values):
    try:
        return [float(v.strip()) for v in values]
    except ValueError:
        return None


def encode(values):
    nums = numeric(values)
    if nums is not None:
        return nums, "numeric"
    if all(v.strip().lower() in ORDER for v in values):
        return [ORDER[v.strip().lower()] for v in values], "ordinal"
    categories = {v.strip(): i + 1 for i, v in enumerate(sorted(set(values)))}
    return [categories[v.strip()] for v in values], "categorical"


def analyze(path, top_n=10):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    effort = [float(row[TARGET].strip()) for row in rows]
    results = []
    for name in rows[0]:
        if name in OUTPUTS:
            continue
        raw = [row[name] for row in rows]
        encoded, kind = encode(raw)
        if len(set(encoded)) < 2:
            continue
        feature_cutoff = sorted(encoded)[len(encoded) // 2]
        high_values = [e for e, value in zip(effort, encoded) if value >= feature_cutoff]
        low_values = [e for e, value in zip(effort, encoded) if value < feature_cutoff]
        if not low_values or not high_values:
            continue
        difference = (sum(high_values) / len(high_values)) - (sum(low_values) / len(low_values))
        score = abs(correlation(encoded, effort))
        results.append({"name": name, "kind": kind, "rho": correlation(encoded, effort),
                        "difference": difference, "levels": len(set(raw)), "score": score})
    return rows, sorted(results, key=lambda item: (-item["score"], -item["difference"], item["name"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    rows, results = analyze(args.csv_path, args.top)
    effort = [float(row[TARGET].strip()) for row in rows]
    print(f"Projects analyzed: {len(rows)}")
    print(f"Effort range: {min(effort):.1f} to {max(effort):.1f} person-months")
    print("Characteristics most associated with high effort (absolute Spearman rank association):")
    for i, item in enumerate(results[:args.top], 1):
        direction = "higher levels tend to increase effort" if item["rho"] > 0 else "higher levels tend to decrease effort"
        print(f"{i}. {item['name']} [{item['kind']}, {item['levels']} levels]: "
              f"rho={item['rho']:+.3f}, high-minus-low effort={item['difference']:+.1f}; {direction}.")
    print("Interpretation: these are associations in this observational dataset, not causal effects.")


if __name__ == "__main__":
    main()
