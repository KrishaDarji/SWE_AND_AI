#!/usr/bin/env python3
"""Categorical NASA93 effort estimator; standard library only."""
import csv, math, random, sys
from collections import Counter, defaultdict

MISSING = "<MISSING>"

def clean(value):
    if value is None or not str(value).strip() or str(value).strip() == "?":
        return MISSING
    return str(value).strip()

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = [x.strip() for x in rows[0]]
    return header, [[clean(x) for x in row] for row in rows[1:] if row]

def predictor_names(header):
    return [name for name in header if not name.endswith("X") and not name.endswith(("+", "-", "!"))]

def grouped_split(rows, features, seed=42, test_fraction=.25):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[tuple(row[j] for j in features)].append(i)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, int(round(len(rows) * test_fraction)))
    test, count = set(), 0
    for key in keys:
        if count < target or not test:
            test.update(groups[key]); count += len(groups[key])
    if len(test) == len(rows):
        key = keys[-1]
        test.difference_update(groups[key])
    train = set(range(len(rows))) - test
    if not train or not test or len(keys) < 2:
        raise ValueError("at least two distinct predictor groups and nonempty train/test sets required")
    return sorted(train), sorted(test)

class CategoricalEstimator:
    """Mode-like symbolic summaries plus exact-group means and Hamming fallback."""
    def __init__(self, names): self.names = list(names); self.fitted = False
    def fit(self, rows, targets):
        vals = [float(x) for x in targets if x != MISSING and math.isfinite(float(x))]
        if len(vals) != len(rows) or not vals: raise ValueError("insufficient finite training targets")
        self.mean = sum(vals) / len(vals); self.n = len(rows)
        self.levels = [set() for _ in self.names]; self.groups = defaultdict(list)
        for row, target in zip(rows, targets):
            key = tuple(clean(x) for x in row)
            for i, x in enumerate(key): self.levels[i].add(x)
            self.groups[key].append(float(target))
        self.modes = [Counter(row[i] for row in rows).most_common(1)[0][0] for i in range(len(self.names))]
        self.fitted = True; return self
    def _vector(self, proposal):
        if not self.fitted: raise ValueError("model is not fitted")
        if set(proposal) != set(self.names): raise ValueError("proposal fields must exactly match predictor schema")
        return tuple(clean(proposal[n]) for n in self.names)
    def predict_one(self, proposal):
        key = self._vector(proposal)
        if key in self.groups: return sum(self.groups[key]) / len(self.groups[key])
        known = [(k, sum(v)/len(v)) for k,v in self.groups.items()]
        scored = sorted((sum(a != b for a,b in zip(key, k)), mean) for k, mean in known)
        best = [mean for distance, mean in scored if distance == scored[0][0]]
        return sum(best) / len(best) if best else self.mean
    def predict(self, proposals): return [self.predict_one(p) for p in proposals]

def mae(actual, predicted): return sum(abs(a-b) for a,b in zip(actual,predicted))/len(actual)

def finite_target(value):
    value = clean(value)
    if value == MISSING: return None
    try: number = float(value)
    except (TypeError, ValueError): return None
    return number if math.isfinite(number) else None

def evaluate(path, proposal=None):
    header, raw = read_csv(path); pnames = predictor_names(header)
    pidx = [header.index(x) for x in pnames]; yidx = header.index("Effort-")
    usable, excluded = [], Counter()
    for row in raw:
        target = finite_target(row[yidx])
        if target is None:
            value = clean(row[yidx]); excluded["missing"] += value == MISSING; excluded["invalid"] += value != MISSING
        else: usable.append((row, target))
    if not usable: raise ValueError("no usable records: Effort- has no finite numeric values")
    rows, ys = [r for r,y in usable], [y for r,y in usable]
    train_i, test_i = grouped_split(rows, pidx)
    model = CategoricalEstimator(pnames).fit([[r[i] for i in pidx] for j,r in enumerate(rows) if j in train_i], [ys[j] for j in train_i])
    train_rows = [[r[i] for i in pidx] for j,r in enumerate(rows) if j in train_i]
    test_rows = [[r[i] for i in pidx] for j,r in enumerate(rows) if j in test_i]
    actual = [ys[j] for j in test_i]
    pred = model.predict([dict(zip(pnames, r)) for r in test_rows])
    baseline = mae(actual, [model.mean] * len(actual))
    result = {"predictors": pnames, "train_count": len(train_i), "test_count": len(test_i), "groups": len(set(tuple(r) for r in train_rows)), "mae": mae(actual,pred), "baseline_mae": baseline, "excluded_targets": dict(excluded), "split": "grouped 75/25, seed=42", "settings": "categorical exact-group mean; Hamming nearest-group fallback; train-only fit"}
    if proposal is not None: result["proposal_effort"] = model.predict_one(proposal)
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2: raise SystemExit("usage: estimator.py DATASET")
    result = evaluate(sys.argv[1]); print("NASA93 effort evaluation")
    for k,v in result.items(): print(f"{k}: {v}")
