"""Independent grouped predictor for X264 encoding performance."""
import csv, math, random


class InputError(ValueError):
    pass


def _finite(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise InputError("nonnumeric value")
    if not math.isfinite(x):
        raise InputError("nonfinite value")
    return x


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not rows[0]:
        raise InputError("empty dataset")
    predictors = [k for k in rows[0] if k != "Performance-"]
    clean = []
    for row in rows:
        try:
            x = tuple(_finite(row[k]) for k in predictors)
            y = _finite(row["Performance-"])
        except (KeyError, InputError):
            continue
        clean.append((x, y))
    if not clean:
        raise InputError("no valid rows")
    return predictors, clean


def grouped_split(rows, seed=42, test_fraction=.2):
    groups = {}
    for x, y in rows:
        groups.setdefault(x, []).append((x, y))
    if len(groups) < 2:
        raise InputError("at least two configuration groups required")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    ntest = max(1, int(round(len(keys) * test_fraction)))
    ntest = min(ntest, len(keys) - 1)
    test_keys = set(keys[:ntest])
    train = [r for k in keys if k not in test_keys for r in groups[k]]
    test = [r for k in keys if k in test_keys for r in groups[k]]
    if not train or not test:
        raise InputError("split produced an empty partition")
    return train, test


def _solve(a, b):
    n = len(b)
    for i in range(n):
        pivot = max(range(i, n), key=lambda j: abs(a[j][i]))
        if abs(a[pivot][i]) < 1e-12:
            a[i][i] += 1e-8
            pivot = i
        a[i], a[pivot] = a[pivot], a[i]; b[i], b[pivot] = b[pivot], b[i]
        scale = a[i][i]
        for j in range(i, n): a[i][j] /= scale
        b[i] /= scale
        for k in range(n):
            if k == i: continue
            factor = a[k][i]
            for j in range(i, n): a[k][j] -= factor * a[i][j]
            b[k] -= factor * b[i]
    return b


class RidgePredictor:
    def __init__(self, names, alpha=1.0):
        self.names, self.alpha = tuple(names), float(alpha)
        self.mean = self.scale = self.coef = self.target_mean = None

    def fit(self, rows):
        if not rows: raise InputError("empty training data")
        p = len(self.names); xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
        self.mean = tuple(sum(x[j] for x in xs) / len(xs) for j in range(p))
        self.scale = tuple(math.sqrt(sum((x[j]-self.mean[j])**2 for x in xs)/len(xs)) or 1.0 for j in range(p))
        z = [[1.0] + [(x[j]-self.mean[j])/self.scale[j] for j in range(p)] for x in xs]
        a = [[sum(r[i]*r[j] for r in z) + (self.alpha if i == j and i else 0.0) for j in range(p+1)] for i in range(p+1)]
        b = [sum(z[k][i]*ys[k] for k in range(len(z))) for i in range(p+1)]
        self.coef = tuple(_solve(a, b)); self.target_mean = sum(ys)/len(ys)
        return self

    def _vector(self, config):
        if set(config) != set(self.names) or len(config) != len(self.names):
            raise InputError("configuration schema must exactly match predictor names")
        return [1.0] + [(_finite(config[n])-self.mean[i])/self.scale[i] for i, n in enumerate(self.names)]

    def predict_one(self, config):
        return sum(a*b for a, b in zip(self.coef, self._vector(config)))

    def predict(self, configs): return [self.predict_one(c) for c in configs]


def mae(model, rows):
    return sum(abs(model.predict_one(dict(zip(model.names, x))) - y) for x, y in rows) / len(rows)


def train_evaluate(path, seed=42):
    names, rows = read_csv(path); train, test = grouped_split(rows, seed)
    model = RidgePredictor(names).fit(train)
    baseline = sum(y for _, y in train) / len(train)
    return {"predictor_names": names, "train_rows": len(train), "test_rows": len(test), "train_groups": len(set(x for x,_ in train)), "test_groups": len(set(x for x,_ in test)), "heldout_mae": mae(model, test), "baseline_mae": sum(abs(y-baseline) for _,y in test)/len(test), "model": model}
