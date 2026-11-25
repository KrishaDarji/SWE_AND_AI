#!/usr/bin/env python3
"""Predict model error and accuracy from a configuration using only stdlib."""
import argparse
import csv
import math
import os
import random
import statistics

FEATURES = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
TARGETS = ("MRE-", "ACC+")


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def encode(row):
    return (float(row[FEATURES[0]]), row[FEATURES[1]], float(row[FEATURES[2]]),
            float(row[FEATURES[3]]), float(row[FEATURES[4]]))


def distance(a, b, scales):
    numeric = (0, 2, 3, 4)
    total = sum(((a[i] - b[i]) / scales[i]) ** 2 for i in numeric)
    total += 1.0 if a[1] != b[1] else 0.0
    return math.sqrt(total)


def fit(rows, k=7):
    points = [encode(row) for row in rows]
    scales = []
    for i in range(5):
        vals = [p[i] for p in points] if i != 1 else []
        scales.append(max(statistics.pstdev(vals), 1e-9) if vals else 1.0)
    return points, scales, k


def predict(model, config):
    points, scales, k = model
    query = encode(config)
    ranked = sorted((distance(query, p, scales), row) for p, row in zip(points, config["_training_rows"]))
    neighbors = ranked[:min(k, len(ranked))]
    weights = [1.0 / (d + 1e-9) for d, _ in neighbors]
    return {target: sum(w * float(row[target]) for w, (_, row) in zip(weights, neighbors)) / sum(weights)
            for target in TARGETS}


def predict_from_rows(training, query, k=7):
    points, scales, k = fit(training, k)
    ranked = sorted(((distance(encode(query), p, scales), row) for p, row in zip(points, training)),
                    key=lambda item: item[0])
    neighbors = ranked[:min(k, len(ranked))]
    weights = [1.0 / (d + 1e-9) for d, _ in neighbors]
    return {target: sum(w * float(row[target]) for w, (_, row) in zip(weights, neighbors)) / sum(weights)
            for target in TARGETS}


def validation_summary(rows, seed=23, fraction=0.05, k=7):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * fraction))
    test, train = shuffled[:cut], shuffled[cut:]
    errors = {target: [] for target in TARGETS}
    for row in test:
        pred = predict_from_rows(train, row, k)
        for target in TARGETS:
            errors[target].append(abs(pred[target] - float(row[target])))
    return {target: statistics.mean(values) for target, values in errors.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--query", type=int, default=0, help="row index to predict")
    args = parser.parse_args()
    rows = load_rows(args.dataset)
    summary = validation_summary(rows)
    query = rows[args.query]
    pred = predict_from_rows(rows[:args.query] + rows[args.query + 1:], query)
    print("validation_mae_mre={:.6f}".format(summary["MRE-"]))
    print("validation_mae_acc={:.6f}".format(summary["ACC+"]))
    print("predicted_error_mre={:.6f}".format(pred["MRE-"]))
    print("predicted_accuracy={:.6f}".format(pred["ACC+"]))


if __name__ == "__main__":
    main()
