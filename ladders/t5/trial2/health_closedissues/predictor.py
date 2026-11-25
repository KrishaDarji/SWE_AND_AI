"""Independent Health configuration predictor; Python standard library only."""
import csv, math, random

PREDICTORS = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
TARGETS = ["MRE-", "ACC+"]
EXCLUDED_OUTCOMES = ["PRED40+"]

def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError): return None

def _key(row):
    vals = []
    for name in PREDICTORS:
        if name == "criterion": vals.append(("cat", str(row.get(name, "")).strip() or "<missing>"))
        else:
            v = _num(row.get(name)); vals.append(("num", v if v is not None else "<missing>"))
    return tuple(vals)

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

class ConfigPredictor:
    """Grouped holdout kNN. Numeric ranges are learned from train rows only."""
    def __init__(self, rows, seed=42, test_fraction=.2):
        self.seed, self.test_fraction = seed, test_fraction
        if not rows: raise ValueError("no rows")
        self.rows = list(rows); self._validate_schema(self.rows[0], require_targets=True)
        groups = {}
        for row in self.rows:
            y = [self._target(row, t) for t in TARGETS]
            if all(v is not None for v in y): groups.setdefault(_key(row), []).append((row, y))
        if len(groups) < 2: raise ValueError("insufficient distinct configuration groups")
        keys = list(groups); random.Random(seed).shuffle(keys)
        ntest = max(1, int(round(len(keys) * test_fraction)))
        ntest = min(ntest, len(keys) - 1)
        testkeys = set(keys[:ntest]); self.train = [x for k in keys if k not in testkeys for x in groups[k]]
        self.test = [x for k in keys if k in testkeys for x in groups[k]]
        if not self.train or not self.test: raise ValueError("empty train/test")
        self.train_keys = {k for k in keys if k not in testkeys}; self.test_keys = testkeys
        self._fit_preprocessing(); self._fit()

    @staticmethod
    def _target(row, name):
        v = _num(row.get(name))
        if v is None: return None
        return v

    def _validate_schema(self, cfg, require_targets=False):
        missing = [x for x in PREDICTORS if x not in cfg]
        allowed = PREDICTORS + TARGETS + (EXCLUDED_OUTCOMES if require_targets else [])
        extra = [x for x in cfg if x not in allowed]
        if missing or extra: raise ValueError("configuration fields invalid: missing=%s extra=%s" % (missing, extra))
        if require_targets:
            bad = [x for x in TARGETS if x not in cfg]
            if bad: raise ValueError("missing target fields: %s" % bad)

    def _fit_preprocessing(self):
        self.ranges = {}
        for name in PREDICTORS:
            if name != "criterion":
                vals = [_num(r.get(name)) for r, _ in self.train]; vals = [v for v in vals if v is not None]
                self.ranges[name] = (min(vals), max(vals)) if vals else (0., 0.)
        self.criteria = sorted({ _key(r)[1][1] for r, _ in self.train })

    def _vector(self, row):
        out = []
        for name in PREDICTORS:
            if name == "criterion": out.extend([1. if _key(row)[1][1] == c else 0. for c in self.criteria])
            else:
                v = _num(row.get(name)); lo, hi = self.ranges[name]
                out.append(.5 if v is None or hi == lo else (v-lo)/(hi-lo))
        return out

    def _fit(self):
        self.mean = {t: sum(y[i] for _, y in self.train)/len(self.train) for i, t in enumerate(TARGETS)}
        self.points = [(self._vector(r), y) for r, y in self.train]

    def predict(self, cfg):
        self._validate_schema(cfg)
        v = self._vector(cfg); scored = []
        for x, y in self.points:
            d = math.sqrt(sum((a-b)**2 for a,b in zip(v,x)))
            scored.append((d,y))
        scored.sort(key=lambda z: z[0]); chosen = scored[:min(7, len(scored))]
        weights = [1/(d+1e-9) for d,_ in chosen]; den = sum(weights)
        return {t: sum(w*y[i] for w,(_,y) in zip(weights,chosen))/den for i,t in enumerate(TARGETS)}

    def evaluate(self):
        got = {t: [] for t in TARGETS}; actual = {t: [] for t in TARGETS}
        for r,y in self.test:
            p = self.predict({k: r[k] for k in PREDICTORS})
            for i,t in enumerate(TARGETS): got[t].append(p[t]); actual[t].append(y[i])
        mae = {t: sum(abs(a-b) for a,b in zip(actual[t],got[t]))/len(got[t]) for t in TARGETS}
        base = {t: sum(abs(a-self.mean[t]) for a in actual[t])/len(actual[t]) for t in TARGETS}
        return {"mae": mae, "baseline_mae": base, "train_rows": len(self.train), "test_rows": len(self.test), "train_groups": len(self.train_keys), "test_groups": len(self.test_keys), "predictors": PREDICTORS}

def fit_from_csv(path, seed=42): return ConfigPredictor(read_csv(path), seed)

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--config", default=None); a = ap.parse_args()
    m = fit_from_csv(a.csv); print(json.dumps(m.evaluate(), sort_keys=True))
    if a.config: print(json.dumps(m.predict(json.loads(a.config)), sort_keys=True))
