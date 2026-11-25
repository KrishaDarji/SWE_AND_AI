#!/usr/bin/env python3
"""NASA93 effort prediction using only the Python standard library."""
import argparse
import csv
import math
from pathlib import Path

ORDINAL = {"vl": 1, "l": 2, "n": 3, "h": 4, "vh": 5, "xh": 6}


def read_projects(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    if not rows or "Effort-" not in rows[0] or "Kloc+" not in rows[0]:
        raise ValueError("CSV must contain Kloc+ and Effort- columns")
    return rows


def features(row):
    values = [1.0, math.log1p(float(row["Kloc+"]))]
    for key, value in row.items():
        if key in {"Effort-", "Kloc+", "idX"}:
            continue
        if key == "YearX":
            values.append((float(value) - 1980.0) / 10.0)
        elif key == "centerX":
            values.append(float(value))
        elif value.strip().lower() in ORDINAL:
            values.append(float(ORDINAL[value.strip().lower()]))
    return values


def solve_linear(matrix, vector, ridge=0.1):
    n = len(vector)
    a = [list(row) + [vector[i]] for i, row in enumerate(matrix)]
    for i in range(n):
        a[i][i] += ridge if i else 0.0
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular design matrix")
        a[col], a[pivot] = a[pivot], a[col]
        divisor = a[col][col]
        a[col] = [x / divisor for x in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor:
                a[row] = [a[row][j] - factor * a[col][j] for j in range(n + 1)]
    return [a[i][n] for i in range(n)]


def fit(rows):
    x = [features(row) for row in rows]
    width = len(x[0])
    if any(len(row) != width for row in x):
        raise ValueError("inconsistent feature dimensions")
    xtx = [[sum(row[i] * row[j] for row in x) for j in range(width)] for i in range(width)]
    xty = [sum(x[k][i] * math.log(float(rows[k]["Effort-"])) for k in range(len(rows))) for i in range(width)]
    return solve_linear(xtx, xty)


def predict(model, row):
    return math.exp(sum(a * b for a, b in zip(model, features(row))))


def leave_one_out(rows):
    errors = []
    for index, row in enumerate(rows):
        model = fit(rows[:index] + rows[index + 1:])
        actual = float(row["Effort-"])
        estimate = predict(model, row)
        errors.append((actual, estimate))
    mae = sum(abs(a - p) for a, p in errors) / len(errors)
    mape = 100.0 * sum(abs(a - p) / a for a, p in errors) / len(errors)
    return mae, mape


def main():
    parser = argparse.ArgumentParser(description="Estimate NASA93 software effort")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--row", type=int, help="1-based project row to predict")
    args = parser.parse_args()
    rows = read_projects(args.csv_path)
    model = fit(rows)
    if args.row:
        if not 1 <= args.row <= len(rows):
            raise SystemExit("row must be between 1 and %d" % len(rows))
        print("project=%d predicted_effort=%.3f person-months" % (args.row, predict(model, rows[args.row - 1])))
    mae, mape = leave_one_out(rows)
    print("projects=%d leave_one_out_mae=%.3f leave_one_out_mape=%.2f%%" % (len(rows), mae, mape))


if __name__ == "__main__":
    main()
