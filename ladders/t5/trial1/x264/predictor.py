#!/usr/bin/env python3
"""Grouped, leakage-safe prediction of X264 encoding performance."""
import csv, math, random

TARGET = "Performance-"

class DataError(ValueError):
    pass

class OnlineNumeric:
    """EZR-inspired Welford summary: count, mean, and variance state."""
    def __init__(self): self.n = 0; self.mean = 0.0; self.m2 = 0.0
    def add(self, x):
        self.n += 1; delta = x - self.mean; self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)
    def sd(self): return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

def _number(value, label):
    try: x = float(value)
    except (TypeError, ValueError): raise DataError("non-numeric " + label)
    if not math.isfinite(x): raise DataError("nonfinite " + label)
    return x

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows: raise DataError("empty dataset")
    names = list(rows[0])
    if TARGET not in names: raise DataError("missing target")
    predictors = [n for n in names if n != TARGET]
    clean = []
    for row_number, row in enumerate(rows, 2):
        try: values = tuple(_number(row[n], n) for n in predictors)
        except DataError as exc: raise DataError("row %d: %s" % (row_number, exc))
        target = _number(row[TARGET], TARGET)
        clean.append((values, target))
    return predictors, clean

def grouped_split(rows, test_fraction=0.2, seed=42):
    groups = {}
    for x, y in rows: groups.setdefault(x, []).append((x, y))
    if len(groups) < 2: raise DataError("at least two distinct configuration groups required")
    keys = list(groups); random.Random(seed).shuffle(keys)
    n_test = max(1, min(len(keys) - 1, round(len(keys) * test_fraction)))
    test_keys = set(keys[:n_test])
    test = [r for k in keys if k in test_keys for r in groups[k]]
    train = [r for k in keys if k not in test_keys for r in groups[k]]
    if not train or not test: raise DataError("empty train/test split")
    return train, test

def _solve(a, b):
    n = len(b); aug = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(aug[i][col]))
        if abs(aug[pivot][col]) < 1e-12: raise DataError("singular model matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        q = aug[col][col]
        aug[col] = [v / q for v in aug[col]]
        for i in range(n):
            if i == col: continue
            q = aug[i][col]
            aug[i] = [u - q * v for u, v in zip(aug[i], aug[col])]
    return [aug[i][-1] for i in range(n)]

class Predictor:
    def __init__(self, names, rows, ridge=1e-6):
        self.names = tuple(names); self.ridge = ridge
        stats = [OnlineNumeric() for _ in names]
        target = OnlineNumeric()
        for x, y in rows:
            for s, v in zip(stats, x): s.add(v)
            target.add(y)
        self.means = tuple(s.mean for s in stats); self.scales = tuple(s.sd() or 1.0 for s in stats)
        self.target_mean = target.mean
        zrows = [[(v-m)/s for v,m,s in zip(x, self.means, self.scales)] for x, _ in rows]
        design = [[1.0] + z for z in zrows]
        a = [[0.0] * len(design[0]) for _ in design[0]]; b = [0.0] * len(a)
        for row, (_, y) in zip(design, rows):
            for i, u in enumerate(row):
                b[i] += u * y
                for j, v in enumerate(row): a[i][j] += u * v
        for i in range(1, len(a)): a[i][i] += ridge
        self.coef = tuple(_solve(a, b)); self.train_count = len(rows)
    def predict(self, config):
        if set(config) != set(self.names) or len(config) != len(self.names):
            raise DataError("prediction configuration schema mismatch")
        x = []
        for name in self.names: x.append((_number(config[name], name)-self.means[self.names.index(name)]) / self.scales[self.names.index(name)])
        return self.coef[0] + sum(c * v for c, v in zip(self.coef[1:], x))

def mae(model, rows): return sum(abs(model.predict(dict(zip(model.names, x))) - y) for x, y in rows) / len(rows)

def fit_evaluate(path):
    names, rows = read_csv(path); train, test = grouped_split(rows)
    model = Predictor(names, train); baseline = sum(y for _, y in train) / len(train)
    return {"predictors": names, "rows": len(rows), "train": len(train), "test": len(test), "groups": len(set(x for x, _ in rows)), "mae": mae(model, test), "baseline_mae": sum(abs(baseline-y) for _, y in test)/len(test), "model": model}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("dataset"); args = p.parse_args()
    r = fit_evaluate(args.dataset)
    print("predictors=" + ",".join(r["predictors"])); print("rows=%d train=%d test=%d groups=%d" % (r["rows"],r["train"],r["test"],r["groups"]))
    print("ridge=1e-6 seed=42 grouped_test_fraction=0.2"); print("held_out_mae=%.9f training_mean_baseline_mae=%.9f" % (r["mae"],r["baseline_mae"]))
