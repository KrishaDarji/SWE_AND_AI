"""Standard-library grouped configuration predictor."""
import csv, math, random

PREDICTORS = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
TARGETS = ("MRE-", "ACC+")

def _number(value):
    if value is None or str(value).strip() in ("", "?"): return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None

def normalize_config(config, strict=True):
    if not isinstance(config, dict) or (strict and set(config) != set(PREDICTORS)):
        raise ValueError("configuration must contain exactly the predictor fields")
    nums = {}
    for name in PREDICTORS:
        if name == "criterion":
            value = config[name]
            nums[name] = "<MISSING>" if value is None or str(value).strip() in ("", "?") else str(value).strip()
        else:
            value = _number(config[name])
            nums[name] = "<MISSING>" if value is None else value
    return tuple(nums[name] for name in PREDICTORS)

def load_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            item = {name: row.get(name) for name in PREDICTORS}
            item.update({target: _number(row.get(target)) for target in TARGETS})
            rows.append(item)
    return rows

def _mean(values):
    return sum(values) / len(values) if values else None

class GuessModel:
    def __init__(self, rows, seed=42):
        if len(rows) < 2: raise ValueError("insufficient rows")
        usable = [r for r in rows if all(r.get(t) is not None for t in TARGETS)]
        groups = {}
        for row in usable: groups.setdefault(normalize_config(row, strict=False), []).append(row)
        if len(groups) < 2: raise ValueError("at least two distinct normalized groups required")
        keys = list(groups); random.Random(seed).shuffle(keys)
        cut = max(1, min(len(keys) - 1, round(len(keys) * .8)))
        self.train_keys, self.test_keys = keys[:cut], keys[cut:]
        self.train = [r for key in self.train_keys for r in groups[key]]
        self.test = [r for key in self.test_keys for r in groups[key]]
        self.train_mean = {t: _mean([r[t] for r in self.train]) for t in TARGETS}
        self.group_mean = {t: {} for t in TARGETS}
        for key in self.train_keys:
            for t in TARGETS: self.group_mean[t][key] = _mean([r[t] for r in groups[key]])
        self.schema = PREDICTORS

    def _predict_one(self, config, target):
        key = normalize_config(config, strict=False)
        return self.group_mean[target].get(key, self.train_mean[target])

    def predict(self, config):
        normalize_config(config)
        return {t: self._predict_one(config, t) for t in TARGETS}

    def evaluate(self):
        scores = {}
        for t in TARGETS:
            predictions = [self._predict_one(r, t) for r in self.test]
            scores[t] = {"heldout_mae": _mean([abs(r[t]-p) for r,p in zip(self.test, predictions)]),
                         "baseline_mae": _mean([abs(r[t]-self.train_mean[t]) for r in self.test])}
        return scores

def fit_csv(path, seed=42): return GuessModel(load_rows(path), seed)
