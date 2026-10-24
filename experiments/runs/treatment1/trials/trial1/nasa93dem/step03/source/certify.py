#!/usr/bin/env python3
"""Certify whether two effort estimators differ beyond sampling noise."""
import argparse
import csv
import math
import random
import statistics
from pathlib import Path

RATINGS = {"vl": 1, "l": 2, "n": 3, "h": 4, "vh": 5, "eh": 6}


def read_projects(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, skipinitialspace=True))


def _rating(value):
    return RATINGS.get(value.strip().lower(), 3.0)


def size_features(row):
    return [1.0, math.log(max(float(row["Kloc+"]), 1e-9))]


def full_features(row):
    fields = ["centerX", "YearX", "prec", "flex", "resl", "team", "pmat",
              "rely", "data", "cplx", "ruse", "docu", "time", "stor",
              "pvol", "acap", "pcap", "pcon", "apex", "plex", "ltex",
              "tool", "site", "sced"]
    values = [1.0, math.log(max(float(row["Kloc+"]), 1e-9)),
              (float(row["YearX"]) - 1980.0) / 20.0]
    for field in fields[2:]:
        values.append(_rating(row[field]))
    return values


def _solve(matrix, target, ridge):
    size = len(matrix[0])
    gram = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    for row, y in zip(matrix, target):
        for i in range(size):
            rhs[i] += row[i] * y
            for j in range(size):
                gram[i][j] += row[i] * row[j]
    for i in range(1, size):
        gram[i][i] += ridge
    for pivot in range(size):
        best = max(range(pivot, size), key=lambda r: abs(gram[r][pivot]))
        if abs(gram[best][pivot]) < 1e-12:
            continue
        gram[pivot], gram[best] = gram[best], gram[pivot]
        rhs[pivot], rhs[best] = rhs[best], rhs[pivot]
        divisor = gram[pivot][pivot]
        gram[pivot] = [v / divisor for v in gram[pivot]]
        rhs[pivot] /= divisor
        for row in range(size):
            if row == pivot:
                continue
            factor = gram[row][pivot]
            gram[row] = [a - factor * b for a, b in zip(gram[row], gram[pivot])]
            rhs[row] -= factor * rhs[pivot]
    return rhs


def fit(rows, kind):
    extractor = size_features if kind == "size" else full_features
    x = [extractor(row) for row in rows]
    y = [math.log(max(float(row["Effort-"]), 1e-9)) for row in rows]
    return extractor, _solve(x, y, 0.5 if kind == "full" else 0.01)


def predict(model, row):
    extractor, coefficients = model
    return math.exp(sum(a * b for a, b in zip(extractor(row), coefficients)))


def folds(rows, repetitions=10, seed=20261008):
    order = list(range(len(rows)))
    rng = random.Random(seed)
    result = []
    for _ in range(repetitions):
        rng.shuffle(order)
        for start in range(10):
            test = order[start::10]
            train = [rows[i] for i in order if i not in test]
            result.append((train, [rows[i] for i in test]))
    return result


def evaluate(rows):
    errors = {"size": [], "full": []}
    for train, test in folds(rows):
        models = {kind: fit(train, kind) for kind in errors}
        for row in test:
            actual = float(row["Effort-"])
            for kind, model in models.items():
                errors[kind].append(abs(predict(model, row) - actual) / actual)
    return errors


def permutation_pvalue(size_errors, full_errors, iterations=10000, seed=20261008):
    observed = statistics.mean(size_errors) - statistics.mean(full_errors)
    rng = random.Random(seed)
    at_least = 0
    for _ in range(iterations):
        signed = [d if rng.random() < 0.5 else -d
                  for d in (a - b for a, b in zip(size_errors, full_errors))]
        if abs(statistics.mean(signed)) >= abs(observed):
            at_least += 1
    return (at_least + 1) / (iterations + 1), observed


def bootstrap_interval(size_errors, full_errors, iterations=5000, seed=20261008):
    differences = [a - b for a, b in zip(size_errors, full_errors)]
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        means.append(statistics.mean(sample))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def certify(path):
    rows = read_projects(path)
    errors = evaluate(rows)
    p_value, difference = permutation_pvalue(errors["size"], errors["full"])
    low, high = bootstrap_interval(errors["size"], errors["full"])
    winner = "full" if difference > 0 else "size"
    meaningful = p_value < 0.05 and low > 0 or p_value < 0.05 and high < 0
    return {"projects": len(rows), "size_mre": statistics.mean(errors["size"]),
            "full_mre": statistics.mean(errors["full"]), "difference": difference,
            "p_value": p_value, "ci_low": low, "ci_high": high,
            "winner": winner, "meaningful": meaningful}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    result = certify(args.dataset)
    print("projects={projects}".format(**result))
    print("size_mre={size_mre:.4f} full_mre={full_mre:.4f}".format(**result))
    print("difference_size_minus_full={difference:.4f}".format(**result))
    print("95pct_ci=[{ci_low:.4f}, {ci_high:.4f}] p_value={p_value:.4f}".format(**result))
    print("meaningfully_more_accurate={meaningful} winner={winner}".format(**result))


if __name__ == "__main__":
    main()
