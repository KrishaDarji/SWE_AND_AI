#!/usr/bin/env python3
"""Estimate project effort from NASA93 historical projects using only stdlib."""
import argparse, csv, json, math, statistics

TARGET = "Effort-"
IGNORE = {"idX", "Kloc+", "Effort-", "Defects-", "MonthsX"}

def load_rows(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))

def number(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None

def distance(a, b, columns, ranges):
    parts = []
    for column in columns:
        left, right = a.get(column, "").strip(), b.get(column, "").strip()
        if left in ("", "?") or right in ("", "?"):
            continue
        x, y = number(left), number(right)
        if x is not None and y is not None:
            parts.append(((x - y) / (ranges[column] or 1.0)) ** 2)
        else:
            parts.append(0.0 if left.lower() == right.lower() else 1.0)
    return math.sqrt(sum(parts) / len(parts)) if parts else float("inf")

def estimate(rows, proposal, k=7):
    columns = [c for c in rows[0] if c not in IGNORE]
    ranges = {}
    for column in columns:
        values = [number(r.get(column)) for r in rows]
        values = [v for v in values if v is not None]
        ranges[column] = max(values) - min(values) if values else 1.0
    ranked = sorted(((distance(proposal, row, columns, ranges), row) for row in rows),
                    key=lambda item: item[0])
    neighbors = [(d, r) for d, r in ranked if math.isfinite(d)][:max(1, min(k, len(ranked)))]
    efforts = [float(r[TARGET]) for _, r in neighbors]
    estimate_value = statistics.mean(efforts)
    spread = statistics.stdev(efforts) if len(efforts) > 1 else 0.0
    return {"estimated_effort_pm": round(estimate_value, 2),
            "uncertainty_pm": round(spread, 2), "neighbors": len(neighbors),
            "nearest_distance": round(neighbors[0][0], 4) if neighbors else None}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("proposal", help="JSON object or path to a JSON object")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--max-effort", type=float, default=100.0)
    parser.add_argument("--max-uncertainty", type=float, default=50.0)
    args = parser.parse_args(argv)
    try:
        with open(args.proposal) as stream:
            proposal = json.load(stream)
    except (OSError, IsADirectoryError):
        proposal = json.loads(args.proposal)
    if isinstance(proposal, str):
        with open(proposal) as stream:
            proposal = json.load(stream)
    result = estimate(load_rows(args.dataset), proposal, args.k)
    result["approve"] = (result["estimated_effort_pm"] <= args.max_effort and
                          result["uncertainty_pm"] <= args.max_uncertainty)
    print(json.dumps(result, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
