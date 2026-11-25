#!/usr/bin/env python3
"""Independent association analysis for the x264 measurements."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score


TARGET = "Performance-"


def load_dataset(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    fields = list(rows[0])
    if TARGET not in fields:
        raise ValueError(f"missing target {TARGET!r}")
    settings = [f for f in fields if f != TARGET]
    if not settings:
        raise ValueError("no settings found")
    matrix = []
    target = []
    for number, row in enumerate(rows, 2):
        try:
            values = [float(row[name]) for name in settings]
            outcome = float(row[TARGET])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric value at CSV row {number}") from exc
        if not all(math.isfinite(v) for v in values + [outcome]):
            raise ValueError(f"non-finite value at CSV row {number}")
        if any(v not in (0.0, 1.0) for v in values):
            raise ValueError(f"setting is not binary at CSV row {number}")
        matrix.append(values)
        target.append(outcome)
    return settings, np.asarray(matrix), np.asarray(target)


def analyze(path: Path, seed: int = 42):
    settings, X, y = load_dataset(path)
    slow_cutoff = float(np.quantile(y, 0.75))
    slow = y >= slow_cutoff
    model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1,
                                  min_samples_leaf=3)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    r2 = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="r2")))
    model.fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=20, random_state=seed,
                                  scoring="r2", n_jobs=1)
    ranking = []
    for i, name in enumerate(settings):
        on = y[X[:, i] == 1]
        off = y[X[:, i] == 0]
        correlation = (float(np.corrcoef(X[:, i], y)[0, 1])
                       if len(on) and len(off) and np.std(X[:, i]) and np.std(y) else 0.0)
        ranking.append({
            "setting": name,
            "spearman_like": correlation,
            "mean_performance_on": float(np.mean(on)) if len(on) else None,
            "mean_performance_off": float(np.mean(off)) if len(off) else None,
            "mean_difference_on_minus_off": float(np.mean(on) - np.mean(off)) if len(on) and len(off) else None,
            "slow_rate_on": float(np.mean(slow[X[:, i] == 1])) if len(on) else None,
            "slow_rate_off": float(np.mean(slow[X[:, i] == 0])) if len(off) else None,
            "permutation_importance_mean": float(perm.importances_mean[i]),
            "permutation_importance_std": float(perm.importances_std[i]),
        })
    ranking.sort(key=lambda item: item["permutation_importance_mean"], reverse=True)
    return {
        "dataset": str(path), "rows": int(len(y)), "settings": len(settings),
        "target": TARGET, "target_mean": float(np.mean(y)),
        "target_min": float(np.min(y)), "target_max": float(np.max(y)),
        "slow_definition": "Performance- >= 75th percentile",
        "slow_cutoff": slow_cutoff, "slow_rows": int(np.sum(slow)),
        "cv_r2_mean": r2, "ranking": ranking,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    report = analyze(args.dataset)
    elapsed = time.perf_counter() - started
    print(f"Loaded {report['rows']} rows, {report['settings']} settings; target={report['target']}")
    print(f"Slow cutoff (75th percentile): {report['slow_cutoff']:.6f}; slow rows={report['slow_rows']}")
    print(f"5-fold random-forest CV R2: {report['cv_r2_mean']:.6f}")
    print("Top settings by permutation importance (mean +/- std):")
    for item in report["ranking"][:10]:
        print(f"  {item['setting']}: {item['permutation_importance_mean']:.6f} +/- {item['permutation_importance_std']:.6f}; "
              f"mean diff={item['mean_difference_on_minus_off']:.6f}; slow-rate diff={item['slow_rate_on']-item['slow_rate_off']:.6f}")
    print(f"Runtime seconds: {elapsed:.6f}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
