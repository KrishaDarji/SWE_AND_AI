#!/usr/bin/env python3
"""Independent association analysis for the x264 measurements."""
import argparse
import csv
import json
import math
import os
from statistics import median

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score


def load_dataset(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    fields = list(rows[0])
    target = next((f for f in fields if f.strip().lower() in {"performance-", "performance", "encoding_time", "encode_time"}), None)
    if target is None:
        raise ValueError("could not identify an encoding-time/performance target")
    features = [f for f in fields if f != target]
    values = []
    for row_number, row in enumerate(rows, 2):
        try:
            values.append([float(row[f].strip()) for f in features] + [float(row[target].strip())])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid numeric value on row {row_number}: {exc}") from exc
    if len(values) < 10:
        raise ValueError("dataset must contain at least 10 observations")
    if any(len(set(row[:-1])) == 1 for row in values):
        # Constant settings are valid, but cannot explain variation.
        pass
    return fields, features, target, values


def spearman(x, y):
    def ranks(a):
        order = sorted(range(len(a)), key=a.__getitem__)
        result = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            rank = (i + j + 2) / 2.0
            for k in range(i, j + 1): result[order[k]] = rank
            i = j + 1
        return result
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def analyze(path, seed=42):
    fields, features, target, rows = load_dataset(path)
    x = [r[:-1] for r in rows]
    y = [r[-1] for r in rows]
    slow_cutoff = median(y)
    slow = [value <= slow_cutoff for value in y]
    correlations = {f: spearman([row[i] for row in x], y) for i, f in enumerate(features)}
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1, min_samples_leaf=2)
    scores = cross_val_score(model, x, y, cv=kfold, scoring="r2")
    model.fit(x, y)
    perm = permutation_importance(model, x, y, n_repeats=30, random_state=seed, scoring="r2", n_jobs=1)
    ranked = []
    for i, feature in enumerate(features):
        ranked.append({"setting": feature, "spearman_rho": correlations[feature],
                       "rf_permutation_mean": perm.importances_mean[i],
                       "rf_permutation_std": perm.importances_std[i],
                       "slow_rate_when_on": sum(v for v, on in zip(slow, [row[i] != 0 for row in x]) if on) / max(1, sum(row[i] != 0 for row in x)),
                       "on_count": sum(row[i] != 0 for row in x)})
    ranked.sort(key=lambda item: (item["rf_permutation_mean"], abs(item["spearman_rho"])), reverse=True)
    return {"dataset": os.path.basename(path), "rows": len(rows), "settings": len(features),
            "target": target, "target_interpretation": "higher Performance- means faster; lower values are slower",
            "slow_definition": f"Performance- <= median ({slow_cutoff:.6g})", "slow_count": sum(slow),
            "performance_min": min(y), "performance_median": slow_cutoff, "performance_max": max(y),
            "cv_r2_mean": sum(scores) / len(scores), "cv_r2_std": math.sqrt(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)),
            "ranked_settings": ranked,
            "limitations": ["Observational measurements establish association, not causation.", "One-hot settings and interactions can confound marginal correlations.", "Permutation importance is sample- and model-dependent; uncertainty is represented by its repeat standard deviation.", "The median threshold is relative to this dataset and is not a universal slow-encode threshold."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    result = analyze(args.dataset)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
