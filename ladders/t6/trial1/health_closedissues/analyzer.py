#!/usr/bin/env python3
"""Step 2 model-configuration quality predictor (standard library only)."""
import csv, math, random, sys
from collections import Counter

CONFIG = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
OUTCOMES = ["MRE-", "ACC+", "PRED40+"]
DIRECTIONS = {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"}
MISSING = {"", "?", "na", "nan", "null", "none"}

def clean(value):
    text = "" if value is None else str(value).strip()
    return None if text.lower() in MISSING else text

def number(value):
    if value is None: return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError): return None

def stats(values):
    xs = sorted(values)
    if not xs: return {"count": 0}
    n = len(xs); q = lambda p: xs[min(n - 1, int(p * n))]
    return {"count": n, "min": xs[0], "max": xs[-1], "mean": sum(xs) / n,
            "median": (xs[(n-1)//2] + xs[n//2]) / 2, "q1": q(.25), "q3": q(.75)}

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        rows = [{(k or "").strip(): clean(v) for k, v in row.items()} for row in reader]
    return headers, rows

def summarize(path):
    headers, rows = read_rows(path)
    if not headers: return {"record_count": 0, "columns": [], "empty": True}
    columns = {}
    for name in headers:
        vals = [r.get(name) for r in rows]; parsed = [number(v) for v in vals if v is not None]
        valid = [x for x in parsed if x is not None]; numeric = len(valid) == len(parsed)
        columns[name] = {"role": "configuration" if name in CONFIG else "outcome" if name in OUTCOMES else "other",
                         "type": "numeric" if numeric else "categorical", "missing": sum(v is None for v in vals)}
        columns[name]["statistics" if numeric else "frequencies"] = stats(valid) if numeric else dict(sorted(Counter(v for v in vals if v is not None).items()))
    counts = Counter(tuple(r.get(k) for k in CONFIG) for r in rows)
    exact = Counter(tuple(r.get(h) for h in headers) for r in rows)
    repeats = {"distinct": len(counts), "repeated_configuration_count": sum(n > 1 for n in counts.values()),
               "repeated_row_count": sum(n for n in counts.values() if n > 1), "duplicate_rows": sum(n-1 for n in counts.values() if n > 1),
               "duplicate_rows_exact": sum(n-1 for n in exact.values() if n > 1)}
    return {"record_count": len(rows), "columns": headers, "column_details": columns, "configuration_fields": CONFIG,
            "outcome_fields": OUTCOMES, "optimization_directions": DIRECTIONS, "configuration_counts": repeats, "empty": not rows}

class _Tree:
    def __init__(self, rows, ys, rng, min_leaf, depth, max_depth):
        self.value = sum(ys) / len(ys); self.left = self.right = None; self.feature = self.cut = None
        if depth >= max_depth or len(rows) < 2 * min_leaf: return
        best = None; base = sum((y-self.value)**2 for y in ys)
        for f in range(len(rows[0])):
            vals = sorted(set(r[f] for r in rows if r[f] is not None))
            if len(vals) > 32: vals = [vals[int(i*(len(vals)-1)/31)] for i in range(32)]
            for a, b in zip(vals, vals[1:]):
                cut = (a+b)/2; li = [i for i,r in enumerate(rows) if r[f] is not None and r[f] <= cut]; ri = [i for i,r in enumerate(rows) if r[f] is not None and r[f] > cut]
                if len(li) < min_leaf or len(ri) < min_leaf: continue
                lm = sum(ys[i] for i in li)/len(li); rm = sum(ys[i] for i in ri)/len(ri)
                gain = base - sum((ys[i]-lm)**2 for i in li) - sum((ys[i]-rm)**2 for i in ri)
                if best is None or gain > best[0]: best = (gain, f, cut, li, ri)
        if best and best[0] > 0:
            _, self.feature, self.cut, li, ri = best
            self.left = _Tree([rows[i] for i in li], [ys[i] for i in li], rng, min_leaf, depth+1, max_depth)
            self.right = _Tree([rows[i] for i in ri], [ys[i] for i in ri], rng, min_leaf, depth+1, max_depth)
    def predict(self, row):
        if self.left is None: return self.value
        return self.left.predict(row) if row[self.feature] <= self.cut else self.right.predict(row)

class ConfigurationPredictor:
    """Grouped 80/20 evaluation and final all-history prediction model."""
    def __init__(self, seed=42, test_fraction=.2, n_trees=5, min_leaf=3, max_depth=4):
        self.seed, self.test_fraction, self.n_trees, self.min_leaf, self.max_depth = seed, test_fraction, n_trees, min_leaf, max_depth
        self.schema = list(CONFIG); self.numeric = [0, 2, 3, 4]; self.trees = {}; self.means = {}; self.preprocessing = {}
    def _fit_preprocessor(self, rows):
        self.preprocessing = {}
        for j, name in enumerate(CONFIG):
            if j in self.numeric:
                xs = [number(r.get(name)) for r in rows]; good = [x for x in xs if x is not None]
                if not good: raise ValueError("no usable training setting: " + name)
                self.preprocessing[name] = ("numeric", sorted(good)[(len(good)-1)//2])
            else:
                good = [clean(r.get(name)) for r in rows]; good = [x for x in good if x is not None]
                if not good: raise ValueError("no usable training setting: " + name)
                self.preprocessing[name] = ("categorical", Counter(good).most_common(1)[0][0], tuple(sorted(set(good))))
    def _x(self, row):
        out = []
        for j, name in enumerate(CONFIG):
            spec = self.preprocessing[name]; v = number(row.get(name)) if j in self.numeric else clean(row.get(name))
            if v is None: v = spec[1]
            if j not in self.numeric: v = float(spec[2].index(v)) if v in spec[2] else -1.0
            out.append(float(v))
        return out
    def fit(self, rows, targets):
        if targets not in ("MRE-", "ACC+"): raise ValueError("target must be MRE- or ACC+")
        usable = [(r, number(r.get(targets))) for r in rows if number(r.get(targets)) is not None]
        groups = {}
        for r, y in usable: groups.setdefault(self.key(r), []).append((r,y))
        if len(groups) < 2: raise ValueError("at least two distinct groups required")
        keys = list(groups); random.Random(self.seed).shuffle(keys); cut = max(1, min(len(keys)-1, round(len(keys)*(1-self.test_fraction))))
        train_keys, test_keys = set(keys[:cut]), set(keys[cut:]); train = [p for k in train_keys for p in groups[k]]; test = [p for k in test_keys for p in groups[k]]
        if not train or not test: raise ValueError("nonempty train/test sets required")
        self._fit_preprocessor([r for r,_ in train]); self.means[targets] = sum(y for _,y in train)/len(train)
        rng = random.Random(self.seed); self.trees[targets] = []
        X, Y = [self._x(r) for r,_ in train], [y for _,y in train]
        for _ in range(self.n_trees):
            sample = [rng.randrange(len(X)) for _ in X]; self.trees[targets].append(_Tree([X[i] for i in sample], [Y[i] for i in sample], rng, self.min_leaf, 0, self.max_depth))
        pred = self.predict_rows([r for r,_ in test], targets); actual = [y for _,y in test]
        mae = sum(abs(a-b) for a,b in zip(actual,pred))/len(actual); base = sum(abs(y-self.means[targets]) for y in actual)/len(actual)
        self.evaluation = {"target": targets, "train_count": len(train), "test_count": len(test), "train_groups": len(train_keys), "test_groups": len(test_keys), "held_out_mae": mae, "baseline_mae": base}
        return self.evaluation
    def key(self, row):
        vals=[]
        for j,n in enumerate(CONFIG):
            v = number(row.get(n)) if j in self.numeric else clean(row.get(n)); vals.append(("missing" if v is None else v) if j not in self.numeric else ("missing" if v is None else float(v)))
        return tuple(vals)
    def predict_rows(self, rows, target):
        if target not in self.trees: raise ValueError("fit target before prediction")
        return [sum(t.predict(self._x(r)) for t in self.trees[target])/len(self.trees[target]) for r in rows]
    def predict(self, configuration, target):
        if set(configuration) != set(CONFIG): raise ValueError("configuration fields must exactly match predictor schema")
        return self.predict_rows([configuration], target)[0]

def main(argv):
    if len(argv) != 2: raise SystemExit("usage: analyzer.py DATASET.csv")
    headers, rows = read_rows(argv[1]); print("Model-quality prediction")
    print("predictors:", ", ".join(CONFIG)); print("outcomes excluded:", ", ".join(OUTCOMES))
    for target in ("MRE-", "ACC+"):
        m=ConfigurationPredictor(); print(target, m.fit(rows, target))
if __name__ == "__main__": main(sys.argv)
