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

def _beta_cf(a, b, x):
    """Continued fraction for the regularized incomplete beta function."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap; d = max(tiny, d); d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = max(tiny, 1 + aa * d); c = max(tiny, 1 + aa / c); d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = max(tiny, 1 + aa * d); c = max(tiny, 1 + aa / c); d = 1 / d
        delta = d * c; h *= delta
        if abs(delta - 1) < 3e-14: break
    return h

def _ibeta(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    front = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x < (a+1)/(a+b+2): return front * _beta_cf(a,b,x) / a
    return 1 - front * _beta_cf(b,a,1-x) / b

def _t_cdf(t, df):
    x = df / (df + t*t); tail = 0.5 * _ibeta(df/2, 0.5, x)
    return 1-tail if t >= 0 else tail

def _t_quantile(p, df):
    lo, hi = -100.0, 100.0
    for _ in range(100):
        mid = (lo+hi)/2
        if _t_cdf(mid, df) < p: lo = mid
        else: hi = mid
    return (lo+hi)/2

def certify(path, threshold):
    """Compare observed Effort- for tool n (A) and h (B), not monetary cost."""
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("practical-effect threshold must be finite and nonnegative")
    _, rows, _ = read_dataset(path)
    groups = {"A": [], "B": []}
    for row in rows:
        tool = row.get("tool", "").strip().lower(); value = _number(row.get("Effort-", ""))
        if tool == "n" and value is not None: groups["A"].append(value)
        elif tool == "h" and value is not None: groups["B"].append(value)
    if len(groups["A"]) < 2 or len(groups["B"]) < 2: raise ValueError("both groups need at least two finite outcomes")
    a, b = groups["A"], groups["B"]; ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b); diff = ma-mb
    se2 = va/len(a)+vb/len(b)
    if se2 == 0: raise ValueError("Welch comparison has zero standard error")
    df = se2*se2/((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1))
    se = math.sqrt(se2); critical = _t_quantile(.975, df); lo, hi = diff-critical*se, diff+critical*se
    p = 2*(1-_t_cdf(abs(diff/se), df)); direction = "B" if diff > 0 else "A" if diff < 0 else "neither"
    return {"groups": {"A": {"tool": "n", "count": len(a), "mean_effort": ma}, "B": {"tool": "h", "count": len(b), "mean_effort": mb}},
            "mean_difference_A_minus_B": diff, "confidence_interval_95": [lo, hi], "welch_df": df, "welch_two_sided_p": p,
            "difference_direction_favors": direction, "threshold": threshold,
            "point_estimate_exceeds_threshold": abs(diff) > threshold,
            "interval_entirely_practically_below_A": hi < -threshold,
            "interval_entirely_practically_below_B": lo > threshold,
            "interpretation": "observed Effort- is not verified monetary cost; practical importance is uncertain if the interval crosses a threshold"}

def flow(path, warmup=2):
    """Evaluate arrivals in CSV order, freezing each estimate before revealing its outcome."""
    header, rows, _ = read_dataset(path)
    predictors = [n for n in header if not n.endswith("X") and n not in OUTCOMES]
    if warmup < 0: raise ValueError("warmup must be nonnegative")
    results, history, scored = [], [], []
    cumulative_model = cumulative_baseline = 0.0
    for index, row in enumerate(rows, 1):
        identifier = row.get("idX", "").strip() or row.get("id", "").strip() or str(index)
        actual = _number(row.get("Effort-", ""))
        if len(history) < warmup:
            prediction = baseline = None
            phase = "warm-up"
        elif not history:
            prediction = baseline = None
            phase = "no-history"
        else:
            model = EffortGuesser(predictors).fit(history)
            prediction = model.predict([row])[0]
            baseline = model.mean
            phase = "scored"
        row_result = {"arrival": index, "identifier": identifier,
                      "prior_training_count": len(history), "phase": phase,
                      "prediction": prediction, "baseline": baseline,
                      "revealed_effort": actual, "absolute_error": None,
                      "baseline_absolute_error": None, "cumulative_mae": None,
                      "baseline_cumulative_mae": None}
        if phase == "scored" and actual is not None:
            error, base_error = abs(prediction - actual), abs(baseline - actual)
            scored.append((error, base_error)); cumulative_model += error; cumulative_baseline += base_error
            row_result.update(absolute_error=error, baseline_absolute_error=base_error,
                              cumulative_mae=cumulative_model / len(scored),
                              baseline_cumulative_mae=cumulative_baseline / len(scored))
        results.append(row_result)
        if actual is not None:
            history.append(row.copy())
    return {"predictors": predictors, "warmup": warmup, "sequence": "CSV order; demonstration sequence, not a verified timeline",
            "records": results, "scored_count": len(scored),
            "mae": cumulative_model / len(scored) if scored else None,
            "baseline_mae": cumulative_baseline / len(scored) if scored else None}

def _effort_groups(rows, descriptor):
    groups = {}
    for row in rows:
        y = _number(row.get("Effort-", ""))
        value = row.get(descriptor, "").strip()
        key = value if value.lower() not in MISSING else "<MISSING>"
        if y is not None: groups.setdefault(key, []).append(y)
    return groups

def blame(path):
    """Rank categorical descriptors by marginal association with observed effort.

    The primary score is eta-squared: between-group sum of squares divided by
    total sum of squares.  This is adapted from EZR's grouped summaries,
    diversity, and discretize/divide ideas; it is descriptive, not causal.
    """
    header, rows, _ = read_dataset(path)
    predictors = [n for n in header if not n.endswith("X") and n not in OUTCOMES]
    ys = [_number(r.get("Effort-", "")) for r in rows]
    finite = [y for y in ys if y is not None]
    overall = statistics.fmean(finite) if finite else None
    total_ss = sum((y - overall) ** 2 for y in finite) if overall is not None else 0.0
    ranked = []
    for order, name in enumerate(predictors):
        groups = _effort_groups(rows, name)
        n = sum(map(len, groups.values()))
        means = {k: statistics.fmean(v) for k, v in groups.items() if v}
        between = sum(len(v) * (means[k] - overall) ** 2 for k, v in groups.items()) if overall is not None else 0.0
        score = between / total_ss if total_ss > 0 else 0.0
        largest = max(means, key=lambda k: (means[k], k)) if means else None
        smallest = min(means, key=lambda k: (means[k], k)) if means else None
        ranked.append({"descriptor": name, "score": score, "direction":
                      "higher in " + largest + "; lower in " + smallest if means and largest != smallest else "no directional contrast",
                      "group_count": len(groups), "usable_count": n,
                      "groups": {k: {"count": len(v), "effort": _stats(v), "mean": means[k]} for k, v in sorted(groups.items())},
                      "note": "marginal categorical association; groups with fewer than 2 observations are retained"})
    ranked.sort(key=lambda x: (-x["score"], predictors.index(x["descriptor"])))
    return {"target": "Effort-", "predictors": predictors, "usable_effort_count": len(finite),
            "overall_effort": _stats(finite), "score": "eta_squared_between_over_total",
            "association_type": "marginal; correlated descriptors can share explanatory information",
            "limitations": ["higher effort alone does not establish inefficiency because scope differs",
                            "delivered size is excluded; this is not size-adjusted productivity",
                            "no causal, defect-risk, process-improvement, or monetary claim"], "ranking": ranked}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (1, 2, 3): raise SystemExit("usage: nasa93_summary.py DATASET.csv [proposal.json] or --certify DATASET.csv THRESHOLD")
    if len(argv) == 2 and argv[0] == "--flow": print(json.dumps(flow(argv[1]), indent=2, sort_keys=True))
    elif len(argv) == 2 and argv[0] == "--blame": print(json.dumps(blame(argv[1]), indent=2, sort_keys=True))
    elif len(argv) == 3 and argv[0] == "--certify": print(json.dumps(certify(argv[1], float(argv[2])), indent=2, sort_keys=True))
    elif len(argv) == 1: print(json.dumps(summarize(argv[0]), indent=2, sort_keys=True))
    else:
        with open(argv[1], encoding="utf-8") as f: proposal = json.load(f)
        print(json.dumps(guess(argv[0], proposal), indent=2, sort_keys=True))

if __name__ == "__main__": main()
