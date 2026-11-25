#!/usr/bin/env python3
"""Choose and justify the best observed x264 configuration."""
import argparse
import csv
import json
from pathlib import Path


def read_measurements(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise ValueError("dataset needs configuration columns and a metric")
        target = reader.fieldnames[-1]
        rows = []
        for number, row in enumerate(reader, 2):
            try:
                row["_metric"] = float(row[target])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid metric on CSV line %d" % number) from exc
            rows.append(row)
    if not rows:
        raise ValueError("dataset is empty")
    return reader.fieldnames[:-1], target, rows


def recommend(path):
    fields, target, rows = read_measurements(path)
    lower_is_better = target.endswith("-")
    chosen = min(rows, key=lambda row: row["_metric"]) if lower_is_better else max(rows, key=lambda row: row["_metric"])
    values = [row["_metric"] for row in rows]
    ordered = sorted(values, reverse=not lower_is_better)
    rank = ordered.index(chosen["_metric"]) + 1
    median = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered)//2-1] + ordered[len(ordered)//2]) / 2
    configuration = {field: chosen[field] for field in fields}
    return {
        "dataset_rows": len(rows),
        "metric": target,
        "direction": "minimize" if lower_is_better else "maximize",
        "recommended_configuration": configuration,
        "observed_metric": chosen["_metric"],
        "rank": rank,
        "median_metric": median,
        "improvement_vs_median_percent": (median - chosen["_metric"]) / median * 100 if lower_is_better else (chosen["_metric"] - median) / median * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Justify an observed x264 configuration.")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    result = recommend(args.dataset)
    print(json.dumps(result, sort_keys=True, indent=2))
    print("Recommendation: deploy the configuration above for validation because it is the best observed row.")
    print("Caveat: this is observational evidence; validate on representative workloads before rollout.")


if __name__ == "__main__":
    main()
