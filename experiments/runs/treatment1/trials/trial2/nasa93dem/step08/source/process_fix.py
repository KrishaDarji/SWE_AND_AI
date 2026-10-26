#!/usr/bin/env python3
"""Recommend a low-cost process change from a NASA93-style CSV."""
import argparse
import csv
import json
import math
import sys


RATING_ORDER = {"vl": 0, "l": 1, "n": 2, "h": 3, "vh": 4, "xh": 5}
TARGETS = ("prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx",
           "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon",
           "apex", "plex", "ltex", "tool", "site", "sced")


def read_rows(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset has no rows")
    required = {"Kloc+", "Defects-", *TARGETS}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("missing columns: " + ", ".join(sorted(missing)))
    for row in rows:
        row["Kloc+"] = float(row["Kloc+"])
        row["Defects-"] = float(row["Defects-"])
    return rows


def risk(row):
    """Defect density, with a small-size guard for malformed records."""
    return row["Defects-"] / max(row["Kloc+"], 0.1)


def recommend(rows):
    baseline = sum(risk(row) for row in rows) / len(rows)
    candidates = []
    for factor in TARGETS:
        levels = sorted({row[factor].strip().lower() for row in rows},
                        key=lambda value: RATING_ORDER.get(value, 99))
        for current in levels:
            if current not in RATING_ORDER or RATING_ORDER[current] == 0:
                continue
            better = RATING_ORDER[current] - 1
            target = next((x for x in levels if RATING_ORDER.get(x) == better), None)
            if target is None:
                continue
            affected = [row for row in rows if row[factor].strip().lower() == current]
            comparable = [row for row in rows if row[factor].strip().lower() == target]
            if not affected or not comparable:
                continue
            current_risk = sum(risk(row) for row in affected) / len(affected)
            target_risk = sum(risk(row) for row in comparable) / len(comparable)
            improvement = current_risk - target_risk
            if improvement <= 0:
                continue
            cost = 1 + (RATING_ORDER[current] - RATING_ORDER[target])
            candidates.append((cost, -improvement, factor, current, target,
                               improvement, len(affected)))
    if not candidates:
        raise ValueError("no process change has a positive estimated risk reduction")
    choice = min(candidates)
    return {
        "baseline_risk": baseline,
        "factor": choice[2], "from": choice[3], "to": choice[4],
        "estimated_risk_reduction": choice[5], "affected_records": choice[6],
        "cost_units": choice[0],
        "method": "empirical defect-density comparison by adjacent rating",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args(argv)
    try:
        result = recommend(read_rows(args.dataset))
    except (OSError, ValueError, csv.Error) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
