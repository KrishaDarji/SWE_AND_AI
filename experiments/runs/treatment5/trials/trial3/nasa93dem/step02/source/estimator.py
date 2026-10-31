"""Categorical grouped estimator for NASA93 Effort- (standard library only)."""
import csv, math, random
from collections import defaultdict

TARGET = "Effort-"
MISSING = "<MISSING>"

def clean(value):
    value = "" if value is None else str(value).strip()
    return value if value else MISSING

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def predictor_names(rows):
    if not rows: raise ValueError("empty dataset")
    return [k for k in rows[0] if not k.endswith("X") and not k.endswith("+") and not k.endswith("-")]

def grouped_split(rows, seed=42, test_fraction=.25):
    names = predictor_names(rows)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[tuple(clean(row.get(k)) for k in names)].append(i)
    if len(groups) < 2: raise ValueError("at least two distinct predictor groups required")
    keys = list(groups); random.Random(seed).shuffle(keys)
    target = max(1, int(round(len(rows) * test_fraction)))
    test, n = [], 0
    for key in keys:
        if test and n >= target: break
        test.extend(groups[key]); n += len(groups[key])
    if not test or len(test) == len(rows): raise ValueError("split produced empty partition")
    testset = set(test)
    return [i for i in range(len(rows)) if i not in testset], test

class CategoricalEstimator:
    def __init__(self, names): self.names = list(names); self.fitted = False
    def _key(self, row): return tuple(clean(row.get(k)) for k in self.names)
    def fit(self, rows):
        if len(rows) < 1: raise ValueError("no training rows")
        values = []
        for row in rows:
            try: y = float(str(row.get(TARGET, "")).strip())
            except (TypeError, ValueError): continue
            if math.isfinite(y): values.append((row, y))
        if not values: raise ValueError("no finite training targets")
        self.mean = sum(y for _, y in values) / len(values)
        self.counts = defaultdict(list)
        self.category = [defaultdict(list) for _ in self.names]
        for row, y in values:
            self.counts[self._key(row)].append(y)
            for j, name in enumerate(self.names): self.category[j][clean(row.get(name))].append(y)
        self.fitted = True; return self
    def predict_one(self, proposal):
        if not self.fitted: raise ValueError("model is not fitted")
        missing = [k for k in self.names if k not in proposal]
        extra = [k for k in proposal if k not in self.names]
        if missing or extra: raise ValueError("proposal schema mismatch: missing=%s extra=%s" % (missing, extra))
        key = self._key(proposal)
        if key in self.counts: return sum(self.counts[key]) / len(self.counts[key])
        vals = []
        for j, name in enumerate(self.names): vals.extend(self.category[j].get(clean(proposal[name]), []))
        return sum(vals) / len(vals) if vals else self.mean
    def predict(self, rows): return [self.predict_one(r) for r in rows]

def mae(actual, predicted): return sum(abs(a-b) for a,b in zip(actual,predicted)) / len(actual)

def evaluate(path, proposal=None):
    rows = load_csv(path); names = predictor_names(rows)
    train_i, test_i = grouped_split(rows)
    train, test = [rows[i] for i in train_i], [rows[i] for i in test_i]
    model = CategoricalEstimator(names).fit(train)
    actual = [float(r[TARGET]) for r in test if r.get(TARGET, "").strip()]
    pred = model.predict([{k: r.get(k) for k in names} for r in test if r.get(TARGET, "").strip()])
    baseline = [model.mean] * len(actual)
    result = {"predictors": names, "train_count": len(train), "test_count": len(test),
              "mae": mae(actual, pred), "baseline_mae": mae(actual, baseline),
              "model_settings": "exact categorical group mean; category fallback; global mean; seed=42"}
    if proposal is not None: result["proposal_prediction"] = model.predict_one(proposal)
    return result
