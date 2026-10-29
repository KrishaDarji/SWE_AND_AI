#!/usr/bin/env python3
"""Independent Treatment 3 race-style selector for the x264 dataset."""
import csv
import json
import random
import time
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DATASET = Path("moot/optimize/config/X264_AllMeasurements.csv")
SEED = 314159
TEST_FRACTION = 0.30
EVAL_BUDGET = 24


def load_dataset(path=DATASET):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "Performance-" not in rows[0]:
        raise ValueError("dataset must contain rows and Performance- target")
    feature_names = [name for name in rows[0] if name != "Performance-"]
    X, y = [], []
    for row in rows:
        try:
            values = [int(row[name].strip()) for name in feature_names]
            target = float(row["Performance-"].strip())
        except (KeyError, ValueError) as exc:
            raise ValueError("non-numeric or missing dataset value") from exc
        if any(value not in (0, 1) for value in values):
            raise ValueError("configuration features must be binary")
        X.append(values)
        y.append(target)
    if len(set(map(tuple, X))) != len(X):
        raise ValueError("configuration rows must be unique")
    return feature_names, X, y


def race_select(train_X, train_y, candidates, budget=EVAL_BUDGET, seed=SEED):
    """Fit a surrogate, then race the predicted best held-out candidates."""
    model = RandomForestRegressor(n_estimators=120, random_state=seed, min_samples_leaf=2)
    model.fit(train_X, train_y)
    ranked = sorted(range(len(candidates)), key=lambda i: model.predict([candidates[i]])[0])
    return ranked[:min(budget, len(ranked))]


def random_search(candidates, budget=EVAL_BUDGET, seed=SEED):
    indices = list(range(len(candidates)))
    random.Random(seed).shuffle(indices)
    return indices[:min(budget, len(indices))]


def evaluate(y, candidate_indices, selected_indices):
    observed = [y[i] for i in selected_indices]
    oracle = min(y[i] for i in candidate_indices)
    best = min(observed)
    return {"best_performance": best, "regret": best - oracle,
            "evaluations": len(selected_indices),
            "oracle_hit": int(best == oracle)}


def run(path=DATASET):
    names, X, y = load_dataset(path)
    train_X, test_X, train_y, test_y, train_i, test_i = train_test_split(
        X, y, list(range(len(y))), test_size=TEST_FRACTION, random_state=SEED)
    selector_local = race_select(train_X, train_y, test_X)
    baseline_local = random_search(test_X)
    selector = evaluate(test_y, range(len(test_y)), selector_local)
    baseline = evaluate(test_y, range(len(test_y)), baseline_local)
    return {"dataset_rows": len(y), "feature_count": len(names), "seed": SEED,
            "test_rows": len(test_y), "evaluation_budget": EVAL_BUDGET,
            "selector": selector, "baseline": baseline,
            "winner_by_regret": "selector" if selector["regret"] < baseline["regret"] else
                                ("baseline" if selector["regret"] > baseline["regret"] else "tie")}


if __name__ == "__main__":
    started = time.perf_counter()
    result = run()
    result["runtime_seconds"] = time.perf_counter() - started
    print(json.dumps(result, sort_keys=True))
