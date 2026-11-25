#!/usr/bin/env python3
"""Predict model error and accuracy from configuration records."""
import argparse
import csv
import json
import math
from pathlib import Path


TARGET_ERROR = "MRE-"
TARGET_ACCURACY = "ACC+"


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _solve(a, b):
    """Gaussian elimination with ridge-like diagonal stabilization."""
    n = len(b)
    m = [list(map(float, row)) + [float(y)] for row, y in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(m[i][col]))
        if abs(m[pivot][col]) < 1e-12:
            m[pivot][col] = 1e-8
        m[col], m[pivot] = m[pivot], m[col]
        divisor = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= divisor
        for i in range(n):
            if i == col:
                continue
            factor = m[i][col]
            if factor:
                for j in range(col, n + 1):
                    m[i][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


class ConfigurationPredictor:
    def __init__(self, rows):
        if not rows:
            raise ValueError("dataset has no rows")
        self.rows = rows
        self.features = [k for k in rows[0] if k not in (TARGET_ERROR, TARGET_ACCURACY, "PRED40+")]
        self.numeric = {k: all(_number(r.get(k)) is not None for r in rows) for k in self.features}
        self.categories = {k: sorted({r.get(k, "") for r in rows}) for k in self.features if not self.numeric[k]}
        self.columns = ["__intercept__"]
        for k in self.features:
            self.columns.append(k)
            if not self.numeric[k]:
                self.columns.extend(k + "=" + c for c in self.categories[k][1:])
        x = [self._vector(r) for r in rows]
        self.models = {}
        for target in (TARGET_ERROR, TARGET_ACCURACY):
            y = [float(r[target]) for r in rows]
            # Small diagonal regularization makes singular designs deterministic.
            xtx = [[sum(row[i] * row[j] for row in x) for j in range(len(self.columns))] for i in range(len(self.columns))]
            for i in range(1, len(xtx)):
                xtx[i][i] += 1e-8
            xty = [sum(row[i] * value for row, value in zip(x, y)) for i in range(len(self.columns))]
            self.models[target] = _solve(xtx, xty)

    def _vector(self, row):
        result = [1.0]
        for k in self.features:
            if self.numeric[k]:
                result.append(float(row.get(k, 0)))
            else:
                result.append(0.0)
                result.extend(1.0 if row.get(k, "") == c else 0.0 for c in self.categories[k][1:])
        return result

    def predict(self, config):
        v = self._vector(config)
        values = {target: sum(a * b for a, b in zip(coef, v)) for target, coef in self.models.items()}
        values[TARGET_ERROR] = max(0.0, min(1.0, values[TARGET_ERROR]))
        values[TARGET_ACCURACY] = max(0.0, min(1.0, values[TARGET_ACCURACY]))
        return {"predicted_error": values[TARGET_ERROR], "predicted_accuracy": values[TARGET_ACCURACY]}


def load(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--config", help="JSON object of configuration fields; defaults to first row")
    args = parser.parse_args(argv)
    rows = load(args.dataset)
    config = json.loads(args.config) if args.config else rows[0]
    print(json.dumps(ConfigurationPredictor(rows).predict(config), sort_keys=True))


if __name__ == "__main__":
    main()
