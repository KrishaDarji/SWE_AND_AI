#!/usr/bin/env python3
"""Suggest a one-setting change for a slow x264 configuration."""
import argparse
import csv
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

DATASET = "/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv"


def load_data(path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("dataset must contain a header and at least one row")
    header = [x.strip() for x in rows[0]]
    features = np.asarray([[int(float(x.strip())) for x in row[:-1]] for row in rows[1:]], dtype=float)
    target = np.asarray([float(row[-1].strip()) for row in rows[1:]], dtype=float)
    if features.shape[1] != len(header) - 1:
        raise ValueError("inconsistent dataset width")
    return header[:-1], features, target


def recommend(path: str = DATASET, config: Dict[str, int] = None) -> Dict[str, object]:
    names, X, y = load_data(path)
    model = RandomForestRegressor(n_estimators=120, random_state=17, n_jobs=1, min_samples_leaf=1)
    model.fit(X, y)
    if config is None:
        index = int(np.argmax(y))
        current = X[index].astype(int)
        source = "slowest_dataset_row"
    else:
        missing = set(config) - set(names)
        if missing:
            raise ValueError("unknown setting(s): " + ", ".join(sorted(missing)))
        current = np.asarray([int(config.get(name, 0)) for name in names], dtype=int)
        index = int(np.argmin(np.sum(np.abs(X - current), axis=1)))
        source = "requested_config"
    current_pred = float(model.predict([current])[0])
    candidates = []
    for col, name in enumerate(names):
        candidate = current.copy()
        candidate[col] = 1 - candidate[col]
        prediction = float(model.predict([candidate])[0])
        candidates.append((prediction, name, candidate))
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_pred, setting, best = candidates[0]
    result = {
        "dataset": os.path.basename(path), "source": source, "setting_changed": setting,
        "from": int(current[names.index(setting)]), "to": int(best[names.index(setting)]),
        "current_performance": current_pred, "predicted_performance": best_pred,
        "predicted_improvement": current_pred - best_pred,
        "configuration": {name: int(value) for name, value in zip(names, best)},
        "measured_source_row": int(index), "feature_count": len(names),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--config", help="JSON object of setting values; omitted uses the slowest row")
    args = parser.parse_args()
    config = json.loads(args.config) if args.config else None
    started = time.perf_counter()
    result = recommend(args.dataset, config)
    result["runtime_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
