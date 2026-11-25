#!/usr/bin/env python3
"""Independent model-guided racing search for the X264 measurements."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def load_measurements(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    target = "Performance-"
    features = [name for name in rows[0] if name != target]
    x = [[int(row[name].strip()) for name in features] for row in rows]
    y = [float(row[target].strip()) for row in rows]
    return x, y, features


def random_baseline(x_test, y_test, budget, seed):
    order = list(range(len(x_test)))
    random.Random(seed).shuffle(order)
    inspected = order[:budget]
    best = min((y_test[i] for i in inspected), default=float("inf"))
    return {"strategy": "random_search", "evaluated": len(inspected), "best": best}


def model_guided_race(x_train, y_train, x_test, y_test, budget, seed):
    """Fit on training measurements, then race predicted test candidates.

    The candidate race uses three increasingly selective rounds. Actual target
    values are revealed only for the survivors; the baseline gets the same
    total evaluation budget and sees candidates in random order.
    """
    model = RandomForestRegressor(n_estimators=80, random_state=seed, n_jobs=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    candidates = list(range(len(x_test)))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    candidates.sort(key=lambda i: predictions[i])
    first = max(1, budget // 2)
    second = max(1, budget // 4)
    final = budget - first - second
    selected = candidates[:max(0, first)]
    if second:
        selected += candidates[first:first + second]
    if final > 0:
        selected += candidates[first + second:first + second + final]
    best = min((y_test[i] for i in selected), default=float("inf"))
    return {"strategy": "model_guided_race", "evaluated": len(selected), "best": best}


def run(dataset, seed=17, test_fraction=0.25, budget=48):
    x, y, features = load_measurements(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_fraction, random_state=seed
    )
    if budget > len(x_test):
        budget = len(x_test)
    race = model_guided_race(x_train, y_train, x_test, y_test, budget, seed)
    baseline = random_baseline(x_test, y_test, budget, seed)
    return {
        "dataset_rows": len(x),
        "feature_count": len(features),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "budget": budget,
        "seed": seed,
        "race": race,
        "baseline": baseline,
        "improvement": baseline["best"] - race["best"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--budget", type=int, default=48)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run(args.dataset, seed=args.seed, budget=args.budget)
    result["runtime_seconds"] = time.perf_counter() - started
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
