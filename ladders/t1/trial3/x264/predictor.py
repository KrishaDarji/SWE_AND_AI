#!/usr/bin/env python3
"""Predict x264 encoding time from historical measurements."""
import argparse
import csv
import json
import math
import sys


def load_measurements(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset has no measurements")
    fields = list(rows[0])
    if "Performance-" not in fields:
        raise ValueError("dataset must contain Performance- target")
    features = [name for name in fields if name != "Performance-"]
    data = []
    for number, row in enumerate(rows, 2):
        try:
            x = [float(row[name].strip()) for name in features]
            y = float(row["Performance-"].strip())
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            raise ValueError("invalid numeric value on CSV row %d" % number) from exc
        data.append((x, y))
    return features, data


def _solve(matrix, vector):
    """Gaussian elimination with partial pivoting."""
    n = len(vector)
    a = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular regression matrix")
        a[col], a[pivot] = a[pivot], a[col]
        divisor = a[col][col]
        a[col] = [value / divisor for value in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor:
                a[row] = [a[row][j] - factor * a[col][j] for j in range(n + 1)]
    return [a[i][-1] for i in range(n)]


class EncodingTimePredictor:
    def __init__(self, features, data, ridge=1e-8):
        self.features = list(features)
        self.mean = sum(y for _, y in data) / len(data)
        size = len(self.features) + 1
        gram = [[0.0] * size for _ in range(size)]
        rhs = [0.0] * size
        for x, y in data:
            row = [1.0] + x
            for i in range(size):
                rhs[i] += row[i] * y
                for j in range(size):
                    gram[i][j] += row[i] * row[j]
        for i in range(1, size):
            gram[i][i] += ridge
        try:
            self.coefficients = _solve(gram, rhs)
        except ValueError:
            self.coefficients = None

    def predict(self, configuration):
        missing = [name for name in self.features if name not in configuration]
        extra = [name for name in configuration if name not in self.features]
        if missing or extra:
            raise ValueError("configuration schema mismatch; missing=%s extra=%s" % (missing, extra))
        try:
            values = [float(configuration[name]) for name in self.features]
        except (TypeError, ValueError) as exc:
            raise ValueError("configuration values must be numeric") from exc
        if self.coefficients is None:
            estimate = self.mean
        else:
            estimate = self.coefficients[0] + sum(c * v for c, v in zip(self.coefficients[1:], values))
        if not math.isfinite(estimate):
            raise ValueError("prediction is not finite")
        return estimate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True, help="JSON object or path to a JSON file")
    args = parser.parse_args(argv)
    features, data = load_measurements(args.dataset)
    predictor = EncodingTimePredictor(features, data)
    config_text = open(args.config, encoding="utf-8").read() if args.config.endswith(".json") else args.config
    configuration = json.loads(config_text)
    prediction = predictor.predict(configuration)
    print(json.dumps({"predicted_encoding_time": prediction, "unit": "dataset target units"}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
