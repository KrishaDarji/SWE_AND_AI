"""Grouped configuration predictor for X264 encoding performance."""
import csv, math, random

TARGET = "Performance-"

def _number(value, name):
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be numeric" % name)
    if not math.isfinite(x):
        raise ValueError("%s must be finite" % name)
    return x

def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or TARGET not in rows[0]: raise ValueError("missing target")
    names = [n for n in rows[0] if n != TARGET]
    data = []
    for row in rows:
        x = tuple(_number(row[n], n) for n in names)
        y = _number(row[TARGET], TARGET)
        data.append((x, y))
    return names, data

def grouped_split(data, test_fraction=.2, seed=42):
    groups = {}
    for x, y in data: groups.setdefault(x, []).append((x, y))
    if len(groups) < 2: raise ValueError("at least two configuration groups required")
    keys = list(groups); random.Random(seed).shuffle(keys)
    count = max(1, min(len(keys)-1, round(len(keys)*test_fraction)))
    test_keys = set(keys[:count])
    train = [r for k in keys if k not in test_keys for r in groups[k]]
    test = [r for k in keys if k in test_keys for r in groups[k]]
    if not train or not test: raise ValueError("empty train/test split")
    return train, test

class Model:
    def __init__(self, names, ridge=1e-8): self.names, self.ridge = list(names), ridge
    def fit(self, rows):
        if not rows: raise ValueError("empty training data")
        p = len(self.names); self.means = [sum(r[0][j] for r in rows)/len(rows) for j in range(p)]
        self.scales = []
        for j in range(p):
            v = sum((r[0][j]-self.means[j])**2 for r in rows)/len(rows)
            self.scales.append(math.sqrt(v) or 1.0)
        z = [[1.0]+[(r[0][j]-self.means[j])/self.scales[j] for j in range(p)] for r in rows]
        targets = [r[1] for r in rows]
        b = [sum(z[k][i] * targets[k] for k in range(len(rows))) for i in range(p+1)]
        a = [[sum(u[i]*u[j] for u in z) for j in range(p+1)] for i in range(p+1)]
        for i in range(1,p+1): a[i][i] += self.ridge
        for i in range(p+1):
            pivot = max(range(i,p+1), key=lambda k: abs(a[k][i]))
            if abs(a[pivot][i]) < 1e-12: raise ValueError("singular training matrix")
            a[i], a[pivot] = a[pivot], a[i]; b[i], b[pivot] = b[pivot], b[i]
            q = a[i][i]; a[i] = [v/q for v in a[i]]; b[i] /= q
            for k in range(p+1):
                if k == i: continue
                q = a[k][i]; a[k] = [a[k][j]-q*a[i][j] for j in range(p+1)]; b[k] -= q*b[i]
        self.coef, self.mean_target = b, sum(targets)/len(targets); return self
    def predict(self, config):
        if set(config) != set(self.names) or len(config) != len(self.names): raise ValueError("configuration schema mismatch")
        x = [_number(config[n], n) for n in self.names]
        return self.coef[0] + sum(self.coef[j+1]*(x[j]-self.means[j])/self.scales[j] for j in range(len(x)))
    def predictions(self, rows): return [self.predict(dict(zip(self.names, x))) for x, _ in rows]

def mae(actual, predicted): return sum(abs(a-p) for a,p in zip(actual,predicted))/len(actual)

def train_evaluate(data, names, seed=42):
    train, test = grouped_split(data, seed=seed); model = Model(names).fit(train)
    pred = model.predictions(test); actual = [y for _,y in test]
    baseline = sum(y for _,y in train)/len(train)
    return model, {"train_rows":len(train), "test_rows":len(test), "train_groups":len(set(x for x,_ in train)), "test_groups":len(set(x for x,_ in test)), "mae":mae(actual,pred), "baseline_mae":mae(actual,[baseline]*len(actual))}
