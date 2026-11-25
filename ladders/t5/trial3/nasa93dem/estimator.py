import csv, math, random
from pathlib import Path

PREDICTORS = ['prec','flex','resl','team','pmat','rely','data','cplx','ruse','docu','time','stor','pvol','acap','pcap','pcon','apex','plex','ltex','tool','site','sced']
TARGET = 'Effort-'
TIE_TOLERANCE = 1e-12
K = 5

def clean(v):
    return '' if v is None else ' '.join(str(v).strip().split())

def finite_float(v):
    try:
        x = float(clean(v))
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def load_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def valid_rows(rows):
    return [r for r in rows if finite_float(r.get(TARGET)) is not None and all(clean(r.get(c)) != '' and clean(r.get(c)) != '?' for c in PREDICTORS)]

def group_key(row):
    return tuple(clean(row.get(c)) for c in PREDICTORS)

class Estimator:
    """Fixed EZR-informed typed-distance k-nearest-neighbor effort estimator."""
    def __init__(self, k=K):
        self.k = k
        self.rows = []
        self.mean = None
        self.numeric = set()
        self.ranges = {}

    def fit(self, rows):
        self.rows = list(rows)
        values = {c: [] for c in PREDICTORS}
        for r in self.rows:
            for c in PREDICTORS:
                x = finite_float(r.get(c))
                if x is not None: values[c].append(x)
        self.numeric = {c for c, xs in values.items() if len(xs) == len(self.rows) and xs}
        self.ranges = {c: (min(values[c]), max(values[c])) for c in self.numeric}
        ys = [finite_float(r[TARGET]) for r in self.rows]
        self.mean = sum(ys) / len(ys)
        return self

    def distance(self, a, b):
        total = 0.0
        for c in PREDICTORS:
            av, bv = clean(a.get(c)), clean(b.get(c))
            if c in self.numeric:
                lo, hi = self.ranges[c]
                scale = hi - lo
                d = 0.0 if scale == 0 else abs(float(av) - float(bv)) / scale
            else:
                d = 0.0 if av == bv else 1.0
            total += d * d
        return math.sqrt(total / len(PREDICTORS))

    def predict_one(self, row):
        ranked = sorted((self.distance(row, r), i, finite_float(r[TARGET])) for i, r in enumerate(self.rows))[:self.k]
        if not ranked: return self.mean
        if ranked[0][0] == 0: return sum(y for d, i, y in ranked if d == 0) / sum(1 for d, i, y in ranked if d == 0)
        weights = [1.0 / d for d, i, y in ranked]
        return sum(w * y for w, (_, _, y) in zip(weights, ranked)) / sum(weights)

    def predict(self, rows): return [self.predict_one(r) for r in rows]

def mae(pred, actual): return sum(abs(a - p) for p, a in zip(pred, actual)) / len(actual)

def split_groups(rows, seed):
    groups = {}
    for i, r in enumerate(rows): groups.setdefault(group_key(r), []).append(i)
    keys = list(groups); random.Random(seed).shuffle(keys)
    target = max(1, round(len(rows) * .25)); test, n = [], 0
    for key in keys:
        if n < target or not test:
            test.extend(groups[key]); n += len(groups[key])
    if len(test) == len(rows):
        key = keys[-1]; test = [i for i in test if i not in groups[key]]
    ts = set(test); train = [i for i in range(len(rows)) if i not in ts]
    return train, test

def evaluate(rows, seed):
    ti, vi = split_groups(rows, seed); train, test = [rows[i] for i in ti], [rows[i] for i in vi]
    model = Estimator().fit(train); pred = model.predict(test)
    actual = [finite_float(r[TARGET]) for r in test]; base = [model.mean] * len(test)
    return {'seed':seed, 'train':train, 'test':test, 'pred':pred, 'actual':actual, 'model_mae':mae(pred, actual), 'baseline_mae':mae(base, actual), 'difference':mae(pred, actual)-mae(base, actual)}
