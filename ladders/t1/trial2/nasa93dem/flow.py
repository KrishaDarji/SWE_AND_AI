#!/usr/bin/env python3
"""Assess incoming software project proposals for portfolio planning."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def _number(row, name):
    return float(row[name].strip())


def load_proposals(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def baseline(rows):
    fields = ("Effort-", "Defects-", "MonthsX")
    values = {field: [_number(row, field) for row in rows] for field in fields}
    return {field: {"median": statistics.median(items), "p75": _percentile(items, .75)}
            for field, items in values.items()}


def _percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def assess(row, stats):
    effort = _number(row, "Effort-")
    defects = _number(row, "Defects-")
    months = _number(row, "MonthsX")
    effort_ratio = effort / stats["Effort-"]["median"]
    defect_ratio = defects / stats["Defects-"]["median"]
    schedule_ratio = months / stats["MonthsX"]["median"]
    risk_score = 100 * (0.45 * min(effort_ratio, 2) / 2 +
                        0.35 * min(defect_ratio, 2) / 2 +
                        0.20 * min(schedule_ratio, 2) / 2)
    if risk_score < 35:
        decision = "prioritize"
    elif risk_score < 65:
        decision = "review"
    else:
        decision = "defer"
    return {
        "id": row.get("idX", ""), "decision": decision,
        "risk_score": round(risk_score, 2),
        "effort": effort, "defects": defects, "months": months,
    }


def assess_stream(rows):
    stats = baseline(rows)
    for row in rows:
        yield assess(row, stats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    rows = load_proposals(args.dataset)
    results = list(assess_stream(rows[:args.limit] if args.limit else rows))
    counts = {name: sum(item["decision"] == name for item in results)
              for name in ("prioritize", "review", "defer")}
    print(json.dumps({"assessed": len(results), "decisions": counts,
                      "proposals": results}, separators=(",", ":")))


if __name__ == "__main__":
    main()
