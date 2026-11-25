#!/usr/bin/env python3
"""One-at-a-time benchmark evaluation for x264 measurements."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


class StreamingEvaluator:
    """Predict and learn one benchmark result at a time."""

    def __init__(self, feature_names, random_state=0):
        self.feature_names = list(feature_names)
        self.scaler = StandardScaler()
        self.model = SGDRegressor(loss="squared_error", penalty="l2",
                                  alpha=1e-4, random_state=random_state,
                                  max_iter=1, tol=None, learning_rate="constant",
                                  eta0=0.01)
        self.seen = 0
        self.best_value = None
        self.best_config = None

    def predict(self, config):
        x = np.asarray([[float(config[name]) for name in self.feature_names]])
        if self.seen == 0:
            return None
        return float(self.model.predict(self.scaler.transform(x))[0])

    def observe(self, config, value):
        x = np.asarray([[float(config[name]) for name in self.feature_names]])
        y = np.asarray([float(value)])
        self.scaler.partial_fit(x)
        scaled = self.scaler.transform(x)
        self.model.partial_fit(scaled, y)
        self.seen += 1
        if self.best_value is None or float(value) < self.best_value:
            self.best_value = float(value)
            self.best_config = {name: int(config[name]) for name in self.feature_names}


def run(path, output=None):
    start = time.perf_counter()
    rows = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Performance-" not in reader.fieldnames:
            raise ValueError("CSV must contain a Performance- column")
        features = [name for name in reader.fieldnames if name != "Performance-"]
        evaluator = StreamingEvaluator(features)
        for index, raw in enumerate(reader, 1):
            config = {name: int(raw[name]) for name in features}
            actual = float(raw["Performance-"])
            predicted = evaluator.predict(config)
            evaluator.observe(config, actual)
            result = {"index": index, "predicted": predicted, "actual": actual,
                      "best_actual": evaluator.best_value}
            rows.append(result)
            if output:
                output.write(json.dumps(result) + "\n")
    return {"rows": len(rows), "features": features,
            "best_actual": evaluator.best_value,
            "best_config": evaluator.best_config,
            "runtime_seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stream = args.output.open("w") if args.output else None
    try:
        summary = run(args.dataset, stream)
    finally:
        if stream:
            stream.close()
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
