#!/usr/bin/env python3
"""Summarize x264 encoder settings and measured performance."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.dataset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
