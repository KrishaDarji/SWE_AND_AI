#!/usr/bin/env python3
"""Recommend a low-cost process improvement from the NASA93 data."""
import argparse
import csv
import json
from pathlib import Path

PROCESS_FIELDS = ("team", "pmat", "tool", "site", "sced")


def load_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, name):
    return float(row[name].strip())


def ordinal(value):
    # NASA93 uses n,l,vl,vh,h and occasional x/blank values.
    names = {"vl": 1, "l": 2, "n": 3, "h": 4, "vh": 5, "x": 3}
    return names.get(value.strip().lower(), 3)


def risk(row):
    kloc = max(number(row, "Kloc+"), 0.001)
    return number(row, "Defects-") / kloc


def recommend(rows):
    if not rows:
        raise ValueError("dataset is empty")
    baseline = sum(risk(row) for row in rows) / len(rows)
    candidates = []
    for field in PROCESS_FIELDS:
        observed = sorted({ordinal(row[field]) for row in rows})
        for level in observed:
            if level <= 1:
                continue
            affected = [row for row in rows if ordinal(row[field]) == level]
            if not affected:
                continue
            # Estimate risk response from the affected group's observed risk.
            group_risk = sum(risk(row) for row in affected) / len(affected)
            improved = baseline - (group_risk * len(affected) / len(rows) * 0.10)
            # Cost proxy: median effort multiplier observed at this level, normalized.
            costs = [number(row, "Effort-") for row in affected]
            cost = sum(costs) / len(costs)
            candidates.append({"field": field, "from_level": level,
                               "to_level": level - 1, "risk": improved,
                               "risk_reduction": baseline - improved, "cost": cost})
    if not candidates:
        return {"baseline_risk": baseline, "recommendation": None}
    best = min(candidates, key=lambda c: (c["cost"], -c["risk_reduction"], c["field"]))
    best["baseline_risk"] = baseline
    return {"baseline_risk": baseline, "recommendation": best}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    result = recommend(load_rows(args.dataset))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
