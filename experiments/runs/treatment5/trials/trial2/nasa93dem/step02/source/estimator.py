"""Independent NASA93 effort estimator using categorical grouped history."""
import csv, math, random

TARGET = "Effort-"
MISSING = "<MISSING>"

def clean(value):
    if value is None:
        return MISSING
    value = str(value).strip()
    return MISSING if value in ("", "?") else value

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return [{k: clean(v) for k, v in row.items()} for row in csv.DictReader(handle)]

def schema(rows):
    if not rows:
        raise ValueError("dataset has no rows")
    names = list(rows[0])
    predictors = [n for n in names if not n.endswith("X") and n not in ("Kloc+", "Effort-", "Defects-") and not n.endswith(("+", "-"))]
    if not predictors:
        raise ValueError("no predictors remain after exclusions")
    return predictors

def valid_rows(rows):
    out = []
    for row in rows:
        try:
            target = float(row[TARGET])
            if math.isfinite(target):
                out.append((row, target))
        except (KeyError, TypeError, ValueError):
            pass
    return out

def grouped_split(rows, predictors, seed=42, test_fraction=.25):
    groups = {}
    for row, target in valid_rows(rows):
        key = tuple(clean(row.get(name)) for name in predictors)
        groups.setdefault(key, []).append((row, target))
    if len(groups) < 2:
        raise ValueError("at least two distinct predictor groups are required")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    wanted = max(1, int(round(len(keys) * test_fraction)))
    wanted = min(wanted, len(keys) - 1)
    test_keys = set(keys[:wanted])
    train = [item for key in keys if key not in test_keys for item in groups[key]]
    test = [item for key in keys if key in test_keys for item in groups[key]]
    if not train or not test:
        raise ValueError("grouped split produced an empty partition")
    return train, test

class Estimator:
    """Smoothed exact-group means with progressively broader categorical backoff."""
    def __init__(self, predictors, smoothing=1.0):
        self.predictors = list(predictors)
        self.smoothing = float(smoothing)
        self.tables = {}
        self.global_mean = None
        self.fitted = False

    def fit(self, pairs):
        if not pairs:
            raise ValueError("cannot fit with no valid training rows")
        self.global_mean = sum(y for _, y in pairs) / len(pairs)
        self.tables = {size: {} for size in range(len(self.predictors), 0, -1)}
        for row, target in pairs:
            vals = tuple(clean(row.get(name)) for name in self.predictors)
            for size in self.tables:
                key = vals[:size]
                count, total = self.tables[size].get(key, (0, 0.0))
                self.tables[size][key] = count + 1, total + target
        self.fitted = True
        return self

    def _check(self, proposal):
        if set(proposal) != set(self.predictors):
            missing = sorted(set(self.predictors) - set(proposal))
            extra = sorted(set(proposal) - set(self.predictors))
            raise ValueError("proposal schema mismatch; missing=%s extra=%s" % (missing, extra))

    def predict_one(self, proposal):
        if not self.fitted:
            raise ValueError("estimator is not fitted")
        self._check(proposal)
        vals = tuple(clean(proposal.get(name)) for name in self.predictors)
        for size in self.tables:
            count_total = self.tables[size].get(vals[:size])
            if count_total:
                count, total = count_total
                return (total + self.smoothing * self.global_mean) / (count + self.smoothing)
        return self.global_mean

    def predict(self, proposals):
        return [self.predict_one(row) for row in proposals]

def mae(actual, predicted):
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)

def evaluate(rows, seed=42):
    predictors = schema(rows)
    train, test = grouped_split(rows, predictors, seed)
    model = Estimator(predictors).fit(train)
    actual = [y for _, y in test]
    predicted = model.predict([{name: r.get(name) for name in predictors} for r, _ in test])
    baseline = sum(y for _, y in train) / len(train)
    return {"model": model, "predictors": predictors, "train": train, "test": test,
            "mae": mae(actual, predicted), "baseline_mae": mae(actual, [baseline] * len(actual))}
