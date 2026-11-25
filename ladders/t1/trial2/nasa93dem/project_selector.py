#!/usr/bin/env python3
"""Choose NASA projects maximizing delivered KLOC under an effort budget."""
import argparse
import csv
import json
import math
from pathlib import Path


def load_projects(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"idX", "Effort-", "Kloc+"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CSV must contain idX, Effort-, and Kloc+")
    projects = []
    for row in rows:
        effort = float(row["Effort-"])
        value = float(row["Kloc+"])
        if effort < 0 or value < 0 or not math.isfinite(effort + value):
            raise ValueError("effort and value must be finite and non-negative")
        projects.append({"id": row["idX"], "effort": effort, "value": value, "settings": row})
    return projects


def choose_projects(projects, budget):
    """Return an exact knapsack solution using cents of effort as integer units."""
    if budget < 0 or not math.isfinite(budget):
        raise ValueError("budget must be finite and non-negative")
    capacity = int(round(budget * 100))
    best = [(0.0, ()) for _ in range(capacity + 1)]
    for index, project in enumerate(projects):
        cost = int(round(project["effort"] * 100))
        if cost > capacity:
            continue
        for remaining in range(capacity, cost - 1, -1):
            prior_value, prior_items = best[remaining - cost]
            candidate = (prior_value + project["value"], prior_items + (index,))
            current = best[remaining]
            if candidate[0] > current[0] + 1e-12 or (abs(candidate[0] - current[0]) <= 1e-12 and candidate[1] < current[1]):
                best[remaining] = candidate
    value, indexes = max(best, key=lambda item: (item[0], -sum(projects[i]["effort"] for i in item[1])))
    selected = [projects[i] for i in indexes]
    return {"budget": budget, "selected": selected, "total_effort": sum(p["effort"] for p in selected), "delivered_kloc": value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--budget", type=float, required=True, help="maximum total Effort-")
    args = parser.parse_args()
    result = choose_projects(load_projects(args.csv_path), args.budget)
    print(json.dumps({"budget": result["budget"], "total_effort": round(result["total_effort"], 2),
                      "delivered_kloc": round(result["delivered_kloc"], 2),
                      "project_ids": [p["id"] for p in result["selected"]],
                      "settings": [p["settings"] for p in result["selected"]]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
