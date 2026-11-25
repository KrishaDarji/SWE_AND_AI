#!/usr/bin/env python3
"""Identify x264 settings associated with slow encodes."""
import argparse
import csv
import json
import time
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score


def load_data(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    target = "Performance-"
    features = [name for name in rows[0] if name != target]
    x = [[float(row[name].strip()) for name in features] for row in rows]
    y = [float(row[target].strip()) for row in rows]
    return features, x, y


def analyze(path, seed=42):
    names, x, y = load_data(path)
    model = RandomForestRegressor(n_estimators=300, random_state=seed, min_samples_leaf=2)
    folds = min(5, len(y))
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, x, y, cv=cv, scoring="neg_mean_absolute_error")
    model.fit(x, y)
    perm = permutation_importance(model, x, y, n_repeats=30, random_state=seed, scoring="neg_mean_absolute_error")
    results = []
    for i, name in enumerate(names):
        active = [y[j] for j, row in enumerate(x) if row[i] == 1]
        inactive = [y[j] for j, row in enumerate(x) if row[i] == 0]
        results.append({
            "setting": name,
            "mean_when_enabled": sum(active) / len(active) if active else None,
            "mean_when_disabled": sum(inactive) / len(inactive) if inactive else None,
            "enabled_minus_disabled": (sum(active) / len(active) - sum(inactive) / len(inactive)) if active and inactive else None,
            "permutation_importance": float(perm.importances_mean[i]),
        })
    results.sort(key=lambda item: (item["enabled_minus_disabled"] is not None, item["enabled_minus_disabled"] or float("-inf")), reverse=True)
    return {
        "rows": len(y), "features": len(names),
        "mean_absolute_error": float(-scores.mean()),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    report = analyze(args.dataset)
    report["runtime_seconds"] = time.perf_counter() - started
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"rows={report['rows']} features={report['features']} CV_MAE={report['mean_absolute_error']:.4f}")
        print("Slow-encode blame (positive enabled_minus_disabled means slower when enabled):")
        for item in report["results"]:
            delta = "n/a" if item["enabled_minus_disabled"] is None else f"{item['enabled_minus_disabled']:.4f}"
            print(f"{item['setting']}: delta={delta} permutation={item['permutation_importance']:.4f}")


if __name__ == "__main__":
    main()
