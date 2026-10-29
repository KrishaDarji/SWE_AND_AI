#!/usr/bin/env python3
"""Summarize x264 encoder settings and measured performance."""
from __future__ import annotations

import argparse
import csv
import json
from time import perf_counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    measure = "Performance-"
    if measure not in rows[0]:
        raise ValueError(f"missing required measure column: {measure}")
    settings = [name for name in rows[0] if name != measure]
    x = np.asarray([[float(row[name].strip()) for name in settings] for row in rows])
    y = np.asarray([float(row[measure].strip()) for row in rows])
    return settings, x, y


def summarize(path: str | Path) -> dict:
    settings, x, y = load_dataset(path)
    scaler = StandardScaler()
    model = LinearRegression().fit(scaler.fit_transform(x), y)
    effects = sorted(
        ({"setting": name, "mean_when_on": float(y[x[:, i] == 1].mean()) if np.any(x[:, i] == 1) else None,
          "mean_when_off": float(y[x[:, i] == 0].mean()) if np.any(x[:, i] == 0) else None,
          "difference_on_minus_off": float(y[x[:, i] == 1].mean() - y[x[:, i] == 0].mean()) if np.any(x[:, i] == 0) and np.any(x[:, i] == 1) else None,
          "standardized_model_coefficient": float(model.coef_[i])}
         for i, name in enumerate(settings)),
        key=lambda item: abs(item["difference_on_minus_off"] or 0), reverse=True)
    best = int(np.argmin(y))
    worst = int(np.argmax(y))
    return {
        "dataset": str(path), "runs": int(len(y)), "setting_count": len(settings),
        "measure": "Performance-", "measure_summary": {
            "mean": float(y.mean()), "std": float(y.std()), "min": float(y.min()),
            "max": float(y.max()), "median": float(np.median(y))},
        "setting_on_counts": {name: int(x[:, i].sum()) for i, name in enumerate(settings)},
        "best_run": {"row": best + 2, "value": float(y[best]),
                     "settings_on": [name for i, name in enumerate(settings) if x[best, i] == 1]},
        "worst_run": {"row": worst + 2, "value": float(y[worst]),
                      "settings_on": [name for i, name in enumerate(settings) if x[worst, i] == 1]},
        "setting_effects": effects,
        "model_r2": float(model.score(scaler.transform(x), y)),
    }


def fit_predictor(x: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit the reusable preprocessing/model pipeline on supplied training data."""
    return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(x, y)


def evaluate(path: str | Path, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Evaluate with a reproducible grouped split; identical configurations stay together."""
    settings, x, y = load_dataset(path)
    groups = np.asarray([",".join(map(str, row)) for row in x])
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(x, y, groups))
    model = fit_predictor(x[train_idx], y[train_idx])
    predictions = model.predict(x[test_idx])
    baseline = np.full(len(test_idx), y[train_idx].mean())
    return {"settings": settings, "train_rows": len(train_idx), "test_rows": len(test_idx),
            "train_groups": len(set(groups[train_idx])), "test_groups": len(set(groups[test_idx])),
            "mae": float(mean_absolute_error(y[test_idx], predictions)),
            "baseline_mae": float(mean_absolute_error(y[test_idx], baseline)),
            "random_state": random_state}


def predict(path: str | Path, configuration: dict[str, float | int]) -> float:
    """Train on all available measurements and predict Performance- for one configuration."""
    settings, x, y = load_dataset(path)
    missing = [name for name in settings if name not in configuration]
    extra = [name for name in configuration if name not in settings]
    if missing or extra:
        raise ValueError(f"configuration keys mismatch; missing={missing}, extra={extra}")
    model = fit_predictor(x, y)
    return float(model.predict(np.asarray([[float(configuration[name]) for name in settings]]))[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--configuration", type=str, help="JSON object of encoder settings")
    args = parser.parse_args()
    started = perf_counter()
    if args.evaluate:
        result = evaluate(args.dataset)
    elif args.configuration:
        result = {"prediction": predict(args.dataset, json.loads(args.configuration))}
    else:
        result = summarize(args.dataset)
    result["runtime_seconds"] = perf_counter() - started
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
