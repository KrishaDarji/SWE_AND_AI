#!/usr/bin/env python3
"""NASA93 descriptive summary, using only the Python standard library."""
import csv, math, statistics, sys, random, json
from collections import Counter

MISSING = {"", "?", "na", "n/a", "null", "none"}
OUTCOMES = {"Kloc+": "maximize", "Effort-": "minimize", "Defects-": "minimize"}

def _number(value):
    text = value.strip()
    if text.lower() in MISSING: return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None

def _kind(values):
    present = [v for v in values if v.strip().lower() not in MISSING]
    return "numeric" if present and all(_number(v) is not None for v in present) else "categorical"

def _stats(values):
    xs = [v for v in values if v is not None]
    if not xs: return {"count": 0, "mean": None, "sd": None, "min": None, "median": None, "max": None}
    return {"count": len(xs), "mean": statistics.fmean(xs),
            "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "median": statistics.median(xs), "max": max(xs)}

def read_dataset(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream, skipinitialspace=False))
    if not rows: return [], [], []
    header = [cell.strip() for cell in rows[0]]
    if not any(header): return [], [], header
    records = []
    for raw in rows[1:]:
        if not raw or all(not cell.strip() for cell in raw): continue
        cells = list(raw[:len(header)]) + [""] * max(0, len(header) - len(raw))
        records.append({name: cells[i].strip() for i, name in enumerate(header)})
    return header, records, rows

def summarize(path):
    header, records, _ = read_dataset(path)
    kinds = {name: ("numeric" if name in OUTCOMES else _kind([r.get(name, "") for r in records])) for name in header}
    roles = {}
    for name in header:
        roles[name] = ("metadata" if name.endswith("X") else
                       "outcome" if name in OUTCOMES else "descriptor")
    numeric, categorical, missing, invalid = {}, {}, {}, {}
    for name in header:
        vals = [r.get(name, "") for r in records]
        missing[name] = sum(v.strip().lower() in MISSING for v in vals)
        if kinds[name] == "numeric":
            parsed = [_number(v) for v in vals]
            numeric[name] = _stats(parsed)
            invalid[name] = sum(v.strip().lower() not in MISSING and _number(v) is None for v in vals)
        else:
            categorical[name] = dict(sorted(Counter(v for v in vals if v.strip().lower() not in MISSING).items()))
            invalid[name] = 0
    return {"record_count": len(records), "columns": header, "types": kinds,
            "roles": roles, "missing": missing, "invalid_numeric": invalid,
            "numeric": numeric, "categorical": categorical,
            "outcomes": {k: {"direction": v, "role": "outcome"} for k, v in OUTCOMES.items() if k in header},
            "model_predictors": [n for n in header if roles[n] == "descriptor"]}

def _group_key(row, predictors):
    return tuple(row.get(n, "").strip() if row.get(n, "").strip().lower() not in MISSING else "<MISSING>" for n in predictors)

class EffortGuesser:
    def __init__(self, predictors, target="Effort-", seed=42):
        self.predictors, self.target, self.seed = list(predictors), target, seed
        self.groups, self.mean, self.fitted = {}, None, False
    def fit(self, rows):
        usable = [(r, _number(r.get(self.target, ""))) for r in rows]
        usable = [(r, y) for r, y in usable if y is not None]
        if not usable: raise ValueError("no finite training targets")
        self.mean = statistics.fmean(y for _, y in usable); totals = {}
        for row, y in usable: totals.setdefault(_group_key(row, self.predictors), []).append(y)
        self.groups = {k: statistics.fmean(v) for k, v in totals.items()}; self.fitted = True
        return self
    def predict(self, rows):
        if not self.fitted: raise ValueError("model is not fitted")
        return [self.groups.get(_group_key(r, self.predictors), self.mean) for r in rows]

def grouped_split(rows, predictors, seed=42, test_fraction=.25):
    groups = {}
    for row in rows: groups.setdefault(_group_key(row, predictors), []).append(row)
    if len(groups) < 2: raise ValueError("at least two distinct predictor groups are required")
    keys = list(groups); random.Random(seed).shuffle(keys); target = max(1, int(round(len(rows)*test_fraction)))
    test, train = [], []
    for key in keys:
        bucket = groups[key]
        if test and len(test) >= target: train.extend(bucket)
        elif test and len(test)+len(bucket) > target: train.extend(bucket)
        else: test.extend(bucket)
    if not train or not test: raise ValueError("split must have nonempty train and test sets")
    return train, test

def guess(path, proposal=None, seed=42):
    header, rows, _ = read_dataset(path)
    predictors = [n for n in header if not n.endswith("X") and n not in OUTCOMES]
    train, test = grouped_split(rows, predictors, seed); model = EffortGuesser(predictors).fit(train)
    actual = [_number(r.get("Effort-", "")) for r in test]; pred = model.predict(test)
    scored = [(a,p) for a,p in zip(actual,pred) if a is not None]
    mae = statistics.fmean(abs(a-p) for a,p in scored) if scored else None
    baseline = statistics.fmean(abs(a-model.mean) for a,p in scored) if scored else None
    result = {"predictors": predictors, "train_count": len(train), "test_count": len(test),
              "groups": len(set(_group_key(r,predictors) for r in rows)), "seed": seed,
              "model": "categorical exact-group mean; unseen/missing groups use training mean",
              "mae": mae, "baseline_mae": baseline, "evaluation": "held-out grouped 75/25 split"}
    if proposal is not None:
        extra, missing = set(proposal)-set(predictors), set(predictors)-set(proposal)
        if extra or missing: raise ValueError("proposal fields must exactly match predictors")
        result["proposal_prediction"] = model.predict([proposal])[0]
    return result

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (1, 2): raise SystemExit("usage: nasa93_summary.py DATASET.csv [proposal.json]")
    if len(argv) == 1: print(json.dumps(summarize(argv[0]), indent=2, sort_keys=True))
    else:
        with open(argv[1], encoding="utf-8") as f: proposal = json.load(f)
        print(json.dumps(guess(argv[0], proposal), indent=2, sort_keys=True))

if __name__ == "__main__": main()
