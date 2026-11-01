"""Grouped, training-only predictor for X264 Performance-."""
import csv, math, random
try:
    from .x264_summary import _number
except ImportError:
    from x264_summary import _number

TARGET = "Performance-"

def remember(store, key, value):
    """Small persistent-in-process memory primitive retained for demonstrations."""
    store[key] = value
    return value

def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as h:
        rows = list(csv.DictReader(h))
    if not rows or not rows[0] or TARGET not in rows[0]:
        raise ValueError("dataset must contain rows and Performance-")
    names = [n for n in rows[0] if n != TARGET]
    clean = []
    for row in rows:
        settings = tuple((row.get(n) or "").strip() for n in names)
        y = _number(row.get(TARGET, ""))
        if all(v not in ("", "?") and _number(v) is not None for v in settings) and y is not None:
            clean.append((settings, y))
    if len({s for s, _ in clean}) < 2:
        raise ValueError("at least two valid configuration groups are required")
    return names, clean

def _split(rows, seed=42, fraction=.8):
    groups = sorted({s for s, _ in rows}, key=repr)
    random.Random(seed).shuffle(groups)
    cut = min(max(1, int(len(groups) * fraction)), len(groups) - 1)
    train_groups = set(groups[:cut])
    train = [r for r in rows if r[0] in train_groups]
    test = [r for r in rows if r[0] not in train_groups]
    return train, test

def _solve(a, b):
    n = len(b); z = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(z[i][col]))
        if abs(z[pivot][col]) < 1e-12: raise ValueError("singular model matrix")
        z[col], z[pivot] = z[pivot], z[col]
        q = z[col][col]
        z[col] = [v / q for v in z[col]]
        for i in range(n):
            if i != col:
                q = z[i][col]
                z[i] = [x - q*y for x, y in zip(z[i], z[col])]
    return [z[i][-1] for i in range(n)]

class Predictor:
    """One-hot linear ridge model; schema and scaling are fitted on training rows."""
    def __init__(self, names, train, ridge=1e-6):
        self.names, self.ridge = list(names), ridge
        self.levels = [sorted({s[i] for s, _ in train}) for i in range(len(names))]
        self.means = []
        raw = [self._raw(s) for s, _ in train]
        for j in range(len(raw[0])):
            self.means.append(sum(x[j] for x in raw) / len(raw))
        self.scales = [math.sqrt(sum((x[j]-self.means[j])**2 for x in raw) / len(raw)) or 1.0 for j in range(len(raw[0]))]
        x = [[(v-m)/q for v, m, q in zip(r, self.means, self.scales)] for r in raw]
        p = len(x[0]); gram = [[0.0]*p for _ in range(p)]; rhs = [0.0]*p
        for r, (_, y) in zip(x, train):
            for i in range(p):
                rhs[i] += r[i]*y
                for j in range(p): gram[i][j] += r[i]*r[j]
        for i in range(p): gram[i][i] += ridge
        self.beta = _solve(gram, rhs)

    def _raw(self, settings):
        out = [1.0]
        for value, levels in zip(settings, self.levels):
            if value not in levels: raise ValueError("configuration contains unseen or invalid setting")
            out.extend(1.0 if value == level else 0.0 for level in levels[1:])
        return out

    def predict(self, configuration):
        if set(configuration) != set(self.names):
            raise ValueError("configuration must contain exactly the fitted predictor names")
        values = [configuration[n] for n in self.names]
        if any(v is None or str(v).strip() in ("", "?") or _number(str(v)) is None for v in values):
            raise ValueError("configuration settings must be finite numeric values")
        raw = self._raw(tuple(str(v).strip() for v in values))
        return sum(b*(v-m)/q for b, v, m, q in zip(self.beta, raw, self.means, self.scales))

def fit_evaluate(path, seed=42):
    names, rows = _read(path); train, test = _split(rows, seed)
    model = Predictor(names, train)
    pred = [model.predict(dict(zip(names, s))) for s, _ in test]
    mae = sum(abs(a-y) for a, (_, y) in zip(pred, test))/len(test)
    mean = sum(y for _, y in train)/len(train)
    baseline = sum(abs(mean-y) for _, y in test)/len(test)
    return model, {"predictors": names, "train_rows": len(train), "test_rows": len(test), "train_groups": len({s for s,_ in train}), "test_groups": len({s for s,_ in test}), "seed": seed, "split": "shuffle unique configurations, 80% groups train", "ridge": model.ridge, "mae": mae, "baseline_mae": baseline}

if __name__ == "__main__":
    import json, sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: x264_predict.py DATASET")
    _, report = fit_evaluate(sys.argv[1])
    print(json.dumps(report, indent=2, sort_keys=True))
