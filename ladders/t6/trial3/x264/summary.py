"""Summarize historical encoder runs using only Python's standard library."""
import argparse
import csv
import json
import math
import statistics
import sys
import random

MISSING = {"", "?", "na", "n/a", "null", "none", "missing"}

class FlowState:
    """Incremental lower-is-better statistics for arriving CSV records."""
    def __init__(self, outcome="Performance-"):
        self.outcome = outcome
        self.total_rows = self.valid_count = self.missing_count = self.invalid_count = 0
        self.mean = 0.0
        self._m2 = 0.0
        self.best = None

    def update(self, row, position):
        self.total_rows += 1
        raw = row.get(self.outcome, "")
        if is_missing(raw):
            self.missing_count += 1
        else:
            try:
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError
            except (TypeError, ValueError):
                self.invalid_count += 1
            else:
                self.valid_count += 1
                delta = value - self.mean
                self.mean += delta / self.valid_count
                self._m2 += delta * (value - self.mean)
                if self.best is None or value < self.best["value"]:
                    settings = {key: val for key, val in row.items() if key != self.outcome}
                    self.best = {"value": value, "position": position, "settings": settings}
        return self.snapshot()

    def snapshot(self):
        return {"total_rows": self.total_rows, "valid_count": self.valid_count,
                "missing_count": self.missing_count, "invalid_count": self.invalid_count,
                "mean": self.mean if self.valid_count else None,
                "sample_variance": self._m2 / (self.valid_count - 1) if self.valid_count > 1 else None,
                "best": None if self.best is None else {
                    "value": self.best["value"], "position": self.best["position"],
                    "settings": dict(self.best["settings"])}}

def flow_rows(rows, outcome="Performance-"):
    """Yield independent snapshots; positions are one-based CSV data rows."""
    state = FlowState(outcome)
    for position, row in enumerate(rows, 1):
        yield state.update(row, position)

def flow_csv(path, outcome="Performance-"):
    """Stream a CSV in file order; the reader and rows are never materialized."""
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return
        names = [name.strip() if name else "" for name in reader.fieldnames]
        if any(not name for name in names):
            raise ValueError("header contains an empty column name")
        state = FlowState(outcome)
        for position, raw in enumerate(reader, 1):
            if None in raw:
                raise ValueError("row has more fields than the header")
            row = {name: (raw.get(name) or "").strip() for name in names}
            yield state.update(row, position)

def flow_final(path, outcome="Performance-"):
    final = None
    for final in flow_csv(path, outcome):
        pass
    return final or FlowState(outcome).snapshot()

def _betacf(a, b, x):
    # Numerical Recipes' continued fraction, with symmetry used by _ibeta.
    tiny, eps, max_iter = 1e-300, 3e-12, 1000
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = max(abs(d), tiny) * (1 if d >= 0 else -1); d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = max(abs(d), tiny) * (1 if d >= 0 else -1); d = 1.0 / d
        c = 1.0 + aa / c; c = max(abs(c), tiny) * (1 if c >= 0 else -1)
        d *= c; h *= d
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = max(abs(d), tiny) * (1 if d >= 0 else -1); d = 1.0 / d
        c = 1.0 + aa / c; c = max(abs(c), tiny) * (1 if c >= 0 else -1)
        delta = d * c; h *= delta
        if abs(delta - 1.0) < eps: return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")

def _ibeta(x, a, b):
    if not 0 <= x <= 1: raise ValueError("beta argument outside [0, 1]")
    if x in (0.0, 1.0): return x
    front = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b) + a*math.log(x) + b*math.log1p(-x))
    if x < (a + 1) / (a + b + 2): return front * _betacf(a,b,x) / a
    return 1.0 - front * _betacf(b,a,1-x) / b

def t_cdf(t, df):
    if df <= 0 or not math.isfinite(t): raise ValueError("finite t and positive degrees of freedom required")
    if t == 0: return 0.5
    sign = 1 if t > 0 else -1; x = abs(t)
    if df > 10000: return 0.5 * (1 + sign * math.erf(x / math.sqrt(2)))
    norm = math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2)) / math.sqrt(df*math.pi)
    n = max(200, int(200*x)+1); n += n % 2; h=x/n
    total = 1.0 + (1+x*x/df) ** (-(df+1)/2)
    for i in range(1,n):
        z=i*h; total += (4 if i%2 else 2) * (1+z*z/df) ** (-(df+1)/2)
    area = norm * h * total / 3
    return .5 + sign * area

def t_quantile(probability, df):
    if not 0 < probability < 1 or df <= 0: raise ValueError("invalid quantile arguments")
    if probability == 0.5: return 0.0
    sign = 1 if probability > .5 else -1
    target = max(probability, 1-probability); lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < target: hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < target: lo = mid
        else: hi = mid
    return sign * (lo + hi) / 2

def certify(path, threshold=None):
    names, rows, warnings = read_runs(path)
    setting = next((n for n in names if n.lower() == "no_8x8dct"), None)
    outcome = next((n for n in names if n.rstrip().lower() == "performance-"), None)
    if setting is None or outcome is None: raise ValueError("required no_8x8dct and Performance- columns are absent")
    if threshold is not None and (not math.isfinite(float(threshold)) or float(threshold) < 0): raise ValueError("threshold must be finite and nonnegative")
    groups = {0: [], 1: []}; excluded = 0
    for row in rows:
        try: g = int(row[setting]); y = float(row[outcome])
        except (KeyError, TypeError, ValueError): excluded += 1; continue
        if g not in groups or not math.isfinite(y): excluded += 1; continue
        groups[g].append(y)
    def stats(xs):
        if not xs: return {"count":0,"mean":None,"sample_variance":None}
        return {"count":len(xs),"mean":statistics.fmean(xs),"sample_variance":statistics.variance(xs) if len(xs)>1 else None}
    result = {"reference":stats(groups[0]),"candidate":stats(groups[1]),"difference":None,"excluded_count":excluded,"outcome":"Performance-","direction":"lower is better","threshold":threshold,"inference":None}
    a,b=groups[0],groups[1]
    if len(a)<2 or len(b)<2: result["inference"]={"defined":False,"reason":"both groups require at least two usable observations"}; return result
    va,vb=statistics.variance(a),statistics.variance(b); ma,mb=statistics.fmean(a),statistics.fmean(b); diff=mb-ma
    se=math.sqrt(va/len(a)+vb/len(b)); result["difference"]=diff
    if se == 0: result["inference"]={"defined":False,"reason":"standard error is zero; inference is undefined"}; return result
    df=(va/len(a)+vb/len(b))**2/((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1))
    t=diff/se; p=2*min(t_cdf(t,df),1-t_cdf(t,df)); crit=t_quantile(.975,df); lo,hi=diff-crit*se,diff+crit*se
    practical = None if threshold is None else diff <= -float(threshold)
    result["difference"] = diff; result["inference"]={"defined":True,"standard_error":se,"t":t,"degrees_of_freedom":df,"p_two_sided":min(1,max(0,p)),"confidence_interval_95":[lo,hi],"statistically_significant":p<.05,"point_meets_practical_threshold":practical,"interval_supports_practical_improvement":None if threshold is None else hi <= -float(threshold),"note":"Welch-Satterthwaite; Simpson integration for the t CDF and bisection quantiles."}
    return result

def is_missing(value):
    return value.strip().lower() in MISSING

def read_runs(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return [], [], ["input has no header"]
        names = [name.strip() if name else "" for name in reader.fieldnames]
        if any(not name for name in names):
            raise ValueError("header contains an empty column name")
        rows = []
        for number, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError("row %d has more fields than the header" % number)
            rows.append({name: (raw.get(name) or "").strip() for name in names})
        return names, rows, []

def numeric_stats(values):
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "median": None, "stdev": None}
    return {"count": len(values), "min": min(values), "max": max(values),
            "mean": statistics.fmean(values), "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}

def summarize(path):
    names, rows, warnings = read_runs(path)
    outcome = next((n for n in names if n.rstrip().lower() == "performance-"), None)
    if outcome is None and names:
        raise ValueError("required measured outcome Performance- is absent")
    settings = [n for n in names if n != outcome]
    frequencies = {n: {} for n in settings}
    invalid = []
    outcome_values = []
    missing_outcomes = 0
    for index, row in enumerate(rows, 2):
        for name in settings:
            value = row[name]
            key = "<missing>" if is_missing(value) else value
            frequencies[name][key] = frequencies[name].get(key, 0) + 1
        value = row.get(outcome, "") if outcome else ""
        if is_missing(value):
            missing_outcomes += 1
            continue
        try:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError
            outcome_values.append(number)
        except ValueError:
            invalid.append({"row": index, "value": value})
    return {"record_count": len(rows), "columns": names, "settings": settings,
            "outcome": {"name": outcome, "direction": "lower is better",
                        "statistics": numeric_stats(outcome_values),
                        "missing_count": missing_outcomes, "invalid": invalid},
            "setting_frequencies": frequencies, "warnings": warnings,
            "assumptions": ["First row is the CSV header.",
                            "Performance- is the sole measured outcome.",
                            "Blank, ?, NA, and common null words are missing."]}

def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None

def _groups(rows, settings):
    groups = {}
    for index, row in enumerate(rows):
        key = tuple(row.get(name, "") for name in settings)
        groups.setdefault(key, []).append(index)
    return groups

def split_groups(rows, settings, seed=42, test_fraction=0.2):
    """Return disjoint row indexes, assigning identical settings together."""
    groups = _groups(rows, settings)
    keys = sorted(groups, key=lambda key: tuple(str(v) for v in key))
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        raise ValueError("at least two configuration groups are required")
    target = max(1, int(round(len(keys) * test_fraction)))
    target = min(target, len(keys) - 1)
    test_keys = set(keys[:target])
    test = sorted(i for key in test_keys for i in groups[key])
    train = sorted(i for key in keys if key not in test_keys for i in groups[key])
    return train, test

class NearestMeanModel:
    """k-nearest mean predictor with train-only numeric ranges and categories."""
    def __init__(self, settings, k=5):
        self.settings, self.k = list(settings), k
        self.numeric, self.categories, self.ranges = {}, {}, {}

    def fit(self, rows, targets):
        if not rows or len(rows) != len(targets):
            raise ValueError("nonempty training data with matching targets is required")
        self.rows, self.targets = list(rows), list(targets)
        for name in self.settings:
            nums = [_finite(row.get(name, "")) for row in self.rows]
            if all(value is not None for value in nums):
                lo, hi = min(nums), max(nums)
                self.numeric[name], self.ranges[name] = True, (lo, hi)
            else:
                self.numeric[name] = False
                self.categories[name] = set(row.get(name, "") for row in self.rows)
        self.mean = statistics.fmean(self.targets)
        return self

    def _distance(self, left, right):
        gaps = []
        for name in self.settings:
            if self.numeric[name]:
                a, b = _finite(left.get(name, "")), _finite(right.get(name, ""))
                lo, hi = self.ranges[name]
                gaps.append(0.5 if a is None or b is None else (0.0 if hi == lo else abs(a-b)/(hi-lo)))
            else:
                gaps.append(0.0 if left.get(name, "") == right.get(name, "") else 1.0)
        return math.sqrt(statistics.fmean([gap * gap for gap in gaps])) if gaps else 0.0

    def predict_one(self, row):
        if set(row) != set(self.settings):
            missing = sorted(set(self.settings) - set(row)); extra = sorted(set(row) - set(self.settings))
            raise ValueError("configuration fields must match; missing=%s extra=%s" % (missing, extra))
        if any(is_missing(str(value)) for value in row.values()):
            raise ValueError("missing or invalid setting value")
        for name in self.settings:
            if self.numeric[name] and _finite(row[name]) is None:
                raise ValueError("invalid numeric setting value for %s" % name)
        nearest = sorted((self._distance(row, old), i) for i, old in enumerate(self.rows))[:max(1, min(self.k, len(self.rows)))]
        return statistics.fmean(self.targets[i] for _, i in nearest)

def evaluate(path, seed=42):
    names, rows, _ = read_runs(path)
    outcome = next((n for n in names if n.rstrip().lower() == "performance-"), None)
    if outcome is None: raise ValueError("required measured outcome Performance- is absent")
    settings = [n for n in names if n != outcome]
    valid = [(i, _finite(row.get(outcome, ""))) for i, row in enumerate(rows)]
    valid = [(i, y) for i, y in valid if y is not None]
    train_i, test_i = split_groups(rows, settings, seed)
    train = [(i, y) for i, y in valid if i in set(train_i)]
    test = [(i, y) for i, y in valid if i in set(test_i)]
    if not train or not test: raise ValueError("finite targets are required in both train and test")
    model = NearestMeanModel(settings).fit([rows[i] for i, _ in train], [y for _, y in train])
    predictions = [model.predict_one({name: rows[i][name] for name in settings}) for i, _ in test]
    mae = statistics.fmean(abs(p-y) for p, (_, y) in zip(predictions, test))
    baseline = statistics.fmean(y for _, y in train)
    base_mae = statistics.fmean(abs(baseline-y) for _, y in test)
    return {"model": model, "settings": settings, "train_rows": len(train), "test_rows": len(test),
            "train_groups": len(set(tuple(rows[i][n] for n in settings) for i, _ in train)),
            "test_groups": len(set(tuple(rows[i][n] for n in settings) for i, _ in test)),
            "held_out_mae": mae, "baseline_mae": base_mae, "training_mean": baseline,
            "excluded_invalid_targets": len(rows)-len(valid), "train_indexes": train_i, "test_indexes": test_i}

def _normalized_setting(value):
    """Normalize numeric spellings so 1, 1.0, and 1e0 share one group."""
    if is_missing(str(value)):
        return None
    number = _finite(value)
    if number is None:
        return None
    return ("number", number)

def _setting_key(value):
    if is_missing(str(value)):
        return None
    numeric = _normalized_setting(value)
    try:
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
    except (TypeError, ValueError):
        pass
    return numeric if numeric is not None else ("text", str(value).strip())

def blame_rows(rows, outcome="Performance-"):
    """Rank settings by marginal eta-squared association with a lower-is-better outcome."""
    if not rows:
        return {"outcome": outcome, "settings": [], "assumptions": _blame_assumptions()}
    names = list(rows[0])
    if outcome not in names:
        raise ValueError("required measured outcome %s is absent" % outcome)
    results = []
    for name in names:
        if name == outcome:
            continue
        groups, excluded = {}, 0
        for row in rows:
            y = _finite(row.get(outcome, ""))
            key = _setting_key(row.get(name, ""))
            if y is None or key is None or key[1] == "":
                excluded += 1
                continue
            groups.setdefault(key, []).append(y)
        eligible = sum(len(values) for values in groups.values())
        means = {key: statistics.fmean(values) for key, values in groups.items()}
        reference = statistics.fmean(y for values in groups.values() for y in values) if eligible else None
        total_ss = sum((y-reference)**2 for values in groups.values() for y in values) if eligible else None
        between_ss = sum(len(values)*(means[key]-reference)**2 for key, values in groups.items()) if eligible else None
        score = between_ss/total_ss if total_ss and len(groups) >= 2 else None
        ordered = sorted(groups, key=lambda key: (means[key], repr(key)))
        result = {"setting": name, "eligible_count": eligible, "excluded_count": excluded,
                  "group_count": len(groups), "groups": [{"value": key[1], "count": len(groups[key]), "mean": means[key]}
                                                           for key in sorted(groups, key=repr)],
                  "reference_mean": reference, "between_group_ss": between_ss, "total_ss": total_ss,
                  "association_score": score, "rank": None,
                  "lower_mean_groups": [key[1] for key in ordered if means[key] < reference] if reference is not None else [],
                  "higher_mean_groups": [key[1] for key in ordered if means[key] > reference] if reference is not None else [],
                  "status": "defined" if score is not None else ("insufficient_data" if eligible < 2 or len(groups) < 2 else "constant_outcome_or_setting")}
        results.append(result)
    results.sort(key=lambda item: (-(item["association_score"] if item["association_score"] is not None else -1), item["setting"]))
    for rank, item in enumerate(results, 1):
        item["rank"] = rank
    return {"outcome": outcome, "direction": "lower is better", "settings": results,
            "assumptions": _blame_assumptions()}

def _blame_assumptions():
    return ["Scores are marginal historical associations, not causal contributions or significance tests.",
            "Missing/nonfinite settings and outcomes are excluded per setting.",
            "Numeric settings are grouped by numeric value; other values by trimmed text.",
            "Undefined scores occur with fewer than two groups or zero total outcome variation."]

def blame(path, outcome="Performance-"):
    names, rows, _ = read_runs(path)
    return blame_rows(rows, outcome)

def _binary_setting(value):
    """Return 0/1 for a complete binary encoder field, otherwise None."""
    if is_missing(str(value)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number not in (0.0, 1.0):
        return None
    return int(number)

def justify_rows(rows, outcome="Performance-"):
    """Select the best eligible measured row and compare it with the median.

    Positions are one-based positions in the supplied data-row sequence.  Blank
    physical CSV records are not emitted by csv.DictReader and are therefore
    not counted; rows supplied directly are counted, including empty mappings.
    """
    if not rows:
        return {"status": "empty_input", "eligible_count": 0, "excluded": [],
                "recommendation": None, "reference": None,
                "associations": blame_rows([], outcome)}
    names = list(rows[0])
    if outcome not in names:
        raise ValueError("required measured outcome %s is absent" % outcome)
    settings = [name for name in names if name != outcome]
    finite_outcomes = []
    eligible = []
    excluded = []
    for position, row in enumerate(rows, 1):
        raw = row.get(outcome, "")
        value = _finite(raw)
        if value is None:
            reason = "missing or nonfinite outcome" if is_missing(str(raw)) else "invalid or nonfinite outcome"
            excluded.append({"position": position, "reason": reason})
            continue
        finite_outcomes.append(value)
        bad = [name for name in settings if _binary_setting(row.get(name, "")) is None]
        if bad:
            excluded.append({"position": position, "reason": "invalid or incomplete binary settings",
                             "fields": bad})
            continue
        configuration = {name: _binary_setting(row[name]) for name in settings}
        eligible.append({"position": position, "value": value, "settings": configuration})
    reference = None if not finite_outcomes else {"count": len(finite_outcomes),
        "median": statistics.median(finite_outcomes)}
    recommendation = min(eligible, key=lambda item: (item["value"], item["position"])) if eligible else None
    comparison = None
    if recommendation is not None and reference is not None:
        reduction = reference["median"] - recommendation["value"]
        comparison = {"absolute_reduction": reduction,
                      "percentage_reduction": None if reference["median"] == 0 else reduction / reference["median"] * 100,
                      "reference": reference}
    status = "ok" if recommendation is not None else ("no_eligible_recommendation" if rows else "empty_input")
    return {"status": status, "selection_basis": "retrospective selection from measured runs; not prediction for an unevaluated configuration",
            "position_convention": "one-based supplied data-row positions; blank physical CSV records are skipped by DictReader",
            "eligible_count": len(eligible), "excluded": excluded,
            "recommendation": recommendation, "comparison": comparison,
            "reference": reference, "associations": blame_rows(rows, outcome),
            "limitations": ["Best only among eligible measured historical configurations.",
                            "Marginal associations are not causal effects, local feature contributions, or proof that changing a setting will improve performance.",
                            "The result does not guarantee future performance."]}

def justify(path, outcome="Performance-"):
    names, rows, warnings = read_runs(path)
    result = justify_rows(rows, outcome)
    result["warnings"] = warnings
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize X264 historical runs")
    parser.add_argument("dataset", help="CSV dataset path")
    parser.add_argument("--blame", action="store_true", help="rank settings by marginal association")
    parser.add_argument("--justify", action="store_true", help="recommend the best eligible measured configuration")
    parser.add_argument("--predict", action="store_true", help="run preserved held-out evaluation")
    parser.add_argument("--certify", action="store_true", help="run preserved group comparison")
    parser.add_argument("--flow", action="store_true", help="run preserved streaming summary")
    args = parser.parse_args(argv)
    try:
        if args.justify: result = justify(args.dataset)
        elif args.blame: result = blame(args.dataset)
        elif args.predict:
            result = evaluate(args.dataset)
            result.pop("model", None)  # preserved API retains it; CLI output is JSON-safe.
        elif args.certify: result = certify(args.dataset)
        elif args.flow: result = flow_final(args.dataset)
        else: result = summarize(args.dataset)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (OSError, ValueError, csv.Error) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
