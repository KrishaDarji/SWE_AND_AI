#!/usr/bin/env python3
"""Step 2 model-configuration quality predictor (standard library only)."""
import csv, json, math, random, sys
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

def _betacf(a,b,x):
    qab,qap,qam=a+b,a+1,a-1; c=1.; d=1-qab*x/qap; d=1e-300 if abs(d)<1e-300 else d; d=1/d; h=d
    for m in range(1,201):
        m2=2*m; aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; d=1e-300 if abs(d)<1e-300 else d; c=1+aa/c; c=1e-300 if abs(c)<1e-300 else c; d=1/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; d=1e-300 if abs(d)<1e-300 else d; c=1+aa/c; c=1e-300 if abs(c)<1e-300 else c; d=1/d; delta=d*c; h*=delta
        if abs(delta-1)<3e-14: break
    return h
def _betai(a,b,x):
    if x<=0:return 0.
    if x>=1:return 1.
    f=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    return f*_betacf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-f*_betacf(b,a,1-x)/b
def _t_cdf(t,df):
    z=df/(df+t*t); q=.5*_betai(df/2,.5,z); return 1-q if t>=0 else q
def _t_quantile(p,df):
    lo,hi=-64.,64.
    for _ in range(100):
        mid=(lo+hi)/2
        if _t_cdf(mid,df)<p:lo=mid
        else:hi=mid
    return (lo+hi)/2
def welch_comparison(rows, practical_threshold=None):
    """Fixed absolute_error minus squared_error comparison for MRE-."""
    if practical_threshold is not None and (not isinstance(practical_threshold,(int,float)) or not math.isfinite(practical_threshold) or practical_threshold<0): raise ValueError("threshold must be finite and nonnegative")
    g={"reference":[],"candidate":[]}
    for row in rows:
        y=number(row.get("MRE-")); c=clean(row.get("criterion"))
        if y is not None and c=="squared_error":g["reference"].append(y)
        elif y is not None and c=="absolute_error":g["candidate"].append(y)
    out={"comparison":"absolute_error - squared_error","outcome":"MRE-","lower_is_better":True,"groups":{k:{"count":len(v),"mean":sum(v)/len(v) if v else None} for k,v in g.items()}}
    r,c=g["reference"],g["candidate"]
    if len(r)<2 or len(c)<2: out.update(status="undefined",reason="each group needs at least two finite outcomes")
    else:
        mr,mc=sum(r)/len(r),sum(c)/len(c); vr=sum((x-mr)**2 for x in r)/(len(r)-1); vc=sum((x-mc)**2 for x in c)/(len(c)-1); se2=vr/len(r)+vc/len(c); d=mc-mr
        if se2==0: out.update(status="undefined",difference=d,reason="standard error is zero; inference undefined")
        else:
            df=se2**2/((vr/len(r))**2/(len(r)-1)+(vc/len(c))**2/(len(c)-1)); se=math.sqrt(se2); t=d/se; p=2*(1-_t_cdf(abs(t),df)); h=_t_quantile(.975,df)*se
            out.update(status="defined",difference=d,standard_error=se,degrees_of_freedom=df,t=t,p_value=p,confidence_interval_95=[d-h,d+h])
    if practical_threshold is None: out["practical"]={"threshold":None,"interpretation":"undetermined without a threshold"}
    else:
        ci=out.get("confidence_interval_95"); point=out.get("difference") is not None and out["difference"]<=-practical_threshold; entire=ci is not None and ci[1]<=-practical_threshold
        out["practical"]={"threshold":practical_threshold,"point_estimate_exceeds":point,"entire_interval_supports":entire,"interpretation":"point estimate exceeds threshold" if point else "point estimate does not exceed threshold"}
    return out

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

class _Running:
    def __init__(self): self.n = 0; self.mean = 0.0; self.m2 = 0.0
    def add(self, x):
        self.n += 1; delta = x - self.mean; self.mean += delta / self.n; self.m2 += delta * (x - self.mean)
    def snapshot(self):
        return {"count": self.n, "mean": self.mean if self.n else None,
                "sample_variance": self.m2 / (self.n - 1) if self.n >= 2 else None}

class Flow:
    """One-pass prefix evaluator; positions are one-based data-row positions."""
    def __init__(self):
        self.rows_seen = 0; self.running = {name: _Running() for name in OUTCOMES}; self.best = {}
    def update(self, row):
        self.rows_seen += 1; position = self.rows_seen
        for outcome in OUTCOMES:
            value = number(row.get(outcome))
            if value is None: continue
            self.running[outcome].add(value)
            old = self.best.get(outcome)
            better = old is None or (value < old["value"] if outcome == "MRE-" else value > old["value"])
            if better:
                self.best[outcome] = {"value": value, "position": position,
                                      "configuration": {key: clean(row.get(key)) for key in CONFIG}}
        return self.snapshot()
    def snapshot(self):
        return {"rows_seen": self.rows_seen,
                "outcomes": {key: value.snapshot() for key, value in self.running.items()},
                "best": {key: {"value": val["value"], "position": val["position"],
                                "configuration": dict(val["configuration"])} for key, val in self.best.items()}}

def flow_rows(rows):
    flow = Flow()
    for row in rows: yield flow.update(row)

def flow_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        yield from flow_rows(({(k or "").strip(): clean(v) for k, v in row.items()} for row in reader))

def _quantile_groups(values, max_groups=4):
    """Return deterministic rank bins; boundaries are setting-only quantiles."""
    unique = sorted(set(values)); k = min(max_groups, len(unique))
    if k <= 1: return {unique[0]: "all"} if unique else {}, []
    groups = {}
    boundaries = []
    for i, value in enumerate(unique):
        group = min(k - 1, (i * k) // len(unique))
        groups[value] = "bin_%d" % (group + 1)
    for i in range(1, k):
        boundaries.append(unique[(i * len(unique) + k - 1) // k - 1])
    return groups, boundaries

def blame(rows, max_numeric_groups=4):
    """Rank marginal configuration associations with finite MRE- (lower is better)."""
    if not isinstance(max_numeric_groups, int) or max_numeric_groups < 1:
        raise ValueError("max_numeric_groups must be a positive integer")
    result = {"score": "eta_squared", "outcome": "MRE-", "lower_is_better": True,
              "numeric_group_rule": "rank bins from sorted unique finite settings, at most %d bins" % max_numeric_groups,
              "fields": [], "status": "defined"}
    for field in CONFIG:
        eligible = []
        for row in rows:
            y = number(row.get("MRE-")); raw = clean(row.get(field))
            x = number(raw) if field != "criterion" else raw
            if y is not None and x is not None: eligible.append((x, y))
        excluded = len(rows) - len(eligible)
        if field == "criterion":
            groups = {x: x for x, _ in eligible}; boundaries = []
        else:
            groups, boundaries = _quantile_groups([x for x, _ in eligible], max_numeric_groups)
        grouped = {}
        for x, y in eligible: grouped.setdefault(groups[x], []).append(y)
        n = len(eligible); mean = sum(y for _, y in eligible) / n if n else None
        between = sum(len(v) * (sum(v) / len(v) - mean) ** 2 for v in grouped.values()) if n else 0.0
        total = sum((y - mean) ** 2 for _, y in eligible) if n else 0.0
        score = between / total if total > 0 else 0.0 if n else None
        summaries = [{"group": g, "count": len(v), "mean_mre": sum(v) / len(v)}
                     for g, v in grouped.items()]
        summaries.sort(key=lambda z: (z["mean_mre"], z["group"]))
        result["fields"].append({"field": field, "eligible_count": n, "excluded_count": excluded,
            "group_count": len(grouped), "reference_mean": mean, "between_ss": between,
            "total_ss": total, "eta_squared": score, "boundaries": boundaries,
            "groups_low_to_high_mre": summaries,
            "status": "empty" if not n else "constant_outcome_or_setting" if not total or len(grouped) < 2 else "defined"})
    result["fields"].sort(key=lambda z: (-1 if z["eta_squared"] is None else -z["eta_squared"], z["field"]))
    for i, field in enumerate(result["fields"], 1): field["rank"] = i
    return result

def _median(values):
    xs = sorted(values); n = len(xs)
    return (xs[(n - 1) // 2] + xs[n // 2]) / 2

def justify(rows):
    """Select the best observed complete row and compare it with outcome medians."""
    rows = list(rows)
    finite_outcomes = {outcome: [number(row.get(outcome)) for row in rows]
                       for outcome in OUTCOMES}
    references = {}
    for outcome, values in finite_outcomes.items():
        good = [value for value in values if value is not None]
        references[outcome] = {"count": len(good), "median": _median(good) if good else None}
    eligible = []
    for position, row in enumerate(rows, 1):
        settings = {}
        valid = True
        for field in CONFIG:
            raw = clean(row.get(field))
            value = raw if field == "criterion" else number(raw)
            if value is None:
                valid = False
            settings[field] = value
        outcomes = {name: number(row.get(name)) for name in OUTCOMES}
        if valid and all(value is not None for value in outcomes.values()):
            eligible.append((outcomes["MRE-"], -outcomes["ACC+"],
                             -outcomes["PRED40+"], position,
                             {"position": position, "configuration": settings,
                              "outcomes": outcomes}))
    result = {"status": "defined" if eligible else ("empty" if not rows else "no_eligible_row"),
              "position_convention": "one-based position among CSV data rows, excluding the header; original positions are retained",
              "selection_rule": ["lowest MRE-", "highest ACC+", "highest PRED40+", "earliest original CSV row position"],
              "eligible_count": len(eligible), "row_count": len(rows),
              "references": references, "direction_labels": DIRECTIONS}
    if not eligible:
        result["recommendation"] = None
        result["context"] = {"status": "unavailable", "reason": "no complete configuration with three finite outcomes"}
        return result
    selected = min(eligible)[4]
    comparisons = {}
    for outcome, value in selected["outcomes"].items():
        reference = references[outcome]
        if reference["median"] is None:
            comparisons[outcome] = {"status": "missing_reference", "selected": value, "reference": reference}
        else:
            delta = value - reference["median"]
            better = delta < 0 if outcome == "MRE-" else delta > 0
            comparisons[outcome] = {"status": "defined", "selected": value,
                                    "reference": reference, "difference": delta,
                                    "direction": DIRECTIONS[outcome], "favorable_vs_median": better}
    association = blame(rows)
    selected_associations = []
    for field_report in association["fields"]:
        field = field_report["field"]
        raw = selected["configuration"][field]
        group = raw if field == "criterion" else next((g for value, g in _quantile_groups(
            [number(row.get(field)) for row in rows if number(row.get(field)) is not None])[0].items() if value == raw), None)
        selected_associations.append({"field": field, "selected_value": raw,
                                      "marginal_eta_squared": field_report["eta_squared"],
                                      "selected_group": group})
    result.update({"recommendation": selected, "comparisons": comparisons,
                   "context": {"status": "defined", "interpretation": "global marginal associations describe the dataset, not causal effects or local contributions", "associations": selected_associations}})
    return result

def choose(rows, cap=100):
    """Retrospectively select an observed feasible configuration under a proxy cap."""
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not math.isfinite(cap) or cap < 0:
        raise ValueError("cap must be finite and nonnegative")
    rows = list(rows); eligible = []; invalid = 0; over_budget = 0
    for position, row in enumerate(rows, 1):
        settings = {field: (clean(row.get(field)) if field == "criterion" else number(row.get(field))) for field in CONFIG}
        outcomes = {field: number(row.get(field)) for field in OUTCOMES}
        valid = (all(settings[field] is not None for field in CONFIG) and
                 all(outcomes[field] is not None for field in OUTCOMES) and
                 settings["N_estimators"] >= 0)
        if not valid:
            invalid += 1
            continue
        if settings["N_estimators"] > cap:
            over_budget += 1
            continue
        eligible.append((outcomes["MRE-"], -outcomes["ACC+"], -outcomes["PRED40+"], position,
                         {"position": position, "configuration": settings, "outcomes": outcomes}))
    result = {"status": "defined" if eligible else ("empty" if not rows else "no_feasible_row"),
              "retrospective": True, "position_convention": "one-based original CSV data-row position",
              "row_count": len(rows), "raw_row_count": len(rows), "valid_row_count": len(rows) - invalid,
              "feasible_row_count": len(eligible), "invalid_exclusion_count": invalid,
              "over_budget_exclusion_count": over_budget, "cap": cap,
              "selected_proxy": "N_estimators", "selection_rule": ["lowest MRE-", "highest ACC+", "highest PRED40+", "earliest original CSV row position"],
              "feasibility": "all five settings and three outcomes finite/complete; nonnegative N_estimators <= cap"}
    result["recommendation"] = min(eligible)[4] if eligible else None
    result["limitations"] = ("N_estimators is a configuration-based compute proxy, not measured runtime or cost; "
                              "other settings may also affect computation. Selection is limited to eligible historical rows "
                              "and does not guarantee performance on unseen evaluations.")
    return result

def _complete(row):
    settings = {field: (clean(row.get(field)) if field == "criterion" else number(row.get(field))) for field in CONFIG}
    outcomes = {field: number(row.get(field)) for field in OUTCOMES}
    return (all(value is not None for value in settings.values()) and
            all(value is not None for value in outcomes.values()), settings, outcomes)

def _changed(old, new):
    return [field for field in CONFIG if old[field] != new[field]]

def recommend_fix(rows):
    """Choose the simplest observed configuration that beats the first row."""
    rows = list(rows)
    result = {"status": "empty" if not rows else None, "row_count": len(rows),
              "retrospective": True,
              "selection_rule": ["fewest changed configuration fields", "lowest MRE-",
                                 "highest ACC+", "highest PRED40+",
                                 "earliest original CSV row position"],
              "position_convention": "one-based original CSV data-row position",
              "candidate_count": 0, "improving_candidate_count": 0,
              "excluded_candidate_count": 0}
    if not rows:
        result.update({"recommendation": None, "reason": "input contains no data rows"})
        return result
    baseline_ok, base_settings, base_outcomes = _complete(rows[0])
    result["baseline"] = {"position": 1, "configuration": base_settings, "outcomes": base_outcomes}
    if not baseline_ok:
        result.update({"status": "invalid_baseline", "recommendation": None,
                       "reason": "first original CSV data row lacks complete finite settings or outcomes",
                       "comparisons": "unavailable/unknown"})
        return result
    candidates = []
    for position, row in enumerate(rows[1:], 2):
        result["candidate_count"] += 1
        ok, settings, outcomes = _complete(row)
        if not ok:
            result["excluded_candidate_count"] += 1
            continue
        if outcomes["MRE-"] < base_outcomes["MRE-"]:
            changed = _changed(base_settings, settings)
            candidates.append((len(changed), outcomes["MRE-"], -outcomes["ACC+"],
                               -outcomes["PRED40+"], position, changed, settings, outcomes))
    result["improving_candidate_count"] = len(candidates)
    if not candidates:
        result.update({"status": "no_improving_candidate", "recommendation": None,
                       "reason": "no complete observed candidate has strictly lower MRE- than baseline"})
        return result
    _, _, _, _, position, changed, settings, outcomes = min(candidates)
    fields = [{"field": field, "old": base_settings[field], "new": settings[field]}
              for field in changed]
    deltas = {name: outcomes[name] - base_outcomes[name] for name in OUTCOMES}
    result.update({"status": "success", "recommendation": {"position": position,
                 "configuration": settings, "outcomes": outcomes,
                 "changed_fields": fields, "changed_field_count": len(fields),
                 "absolute_mre_improvement": base_outcomes["MRE-"] - outcomes["MRE-"],
                 "secondary_outcome_differences": {"ACC+": deltas["ACC+"],
                                                     "PRED40+": deltas["PRED40+"]}},
                 "limitations": "This is a retrospective comparison between observed rows. Change count is a simplicity proxy; historical differences do not prove causation or guarantee future improvement."})
    return result

def _mad(values, center):
    return _median([abs(x - center) for x in values])

def spot(rows, threshold=3.5, min_sample=3):
    """Screen marginal outcomes using self-inclusive median/MAD modified z-scores.

    Score is 0.67448975*(x-median)/MAD; flag abs(score)>threshold.
    Each finite observation contributes to its metric's reference distribution.
    With fewer than min_sample values, scores are undefined. With MAD zero,
    equal-to-center values have score 0 and non-center values are undefined.
    """
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if not isinstance(min_sample, int) or min_sample < 1: raise ValueError("min_sample must be positive")
    rows = list(rows); reports = {}; flagged = []
    for outcome in OUTCOMES:
        finite = [(pos, row, number(row.get(outcome))) for pos, row in enumerate(rows, 1)]
        finite = [(p, r, x) for p, r, x in finite if x is not None]
        center = _median([x for _, _, x in finite]) if finite else None
        scale = _mad([x for _, _, x in finite], center) if finite else None
        info = {"reference_count": len(finite), "center": center, "mad": scale,
                "threshold": threshold, "minimum_sample": min_sample,
                "reference_includes_observation": True, "flags": [],
                "undefined_score_count": 0, "missing_or_nonfinite_count": len(rows)-len(finite)}
        for pos, row, value in finite:
            score = None; reason = None
            if len(finite) < min_sample: reason = "undefined: reference sample is shorter than minimum"
            elif scale == 0:
                if value == center: score = 0.0
                else: reason = "undefined: zero MAD for non-center observation"
            else: score = 0.67448975 * (value - center) / scale
            if score is None:
                info["undefined_score_count"] += 1
            elif abs(score) > threshold:
                item = {"position": pos, "configuration": {k: clean(row.get(k)) for k in CONFIG},
                        "outcome": outcome, "observed": value, "score": score,
                        "reference": {"count": len(finite), "center": center, "mad": scale,
                                      "threshold": threshold},
                        "reason": "absolute modified MAD score exceeds threshold"}
                info["flags"].append(item); flagged.append(item)
        reports[outcome] = info
    return {"method": "self-inclusive median/MAD modified z-score", "reports": reports,
            "per_outcome_flag_counts": {o: len(reports[o]["flags"]) for o in OUTCOMES},
            "unique_flagged_row_count": len({x["position"] for x in flagged}),
            "flags": flagged,
            "interpretation": "Unusual results warrant investigation, not proof of measurement error; very good results can also be unusual. This is marginal outcome screening, not configuration-conditioned anomaly detection or a validated error classifier."}

def main(argv):
    if len(argv) != 2: raise SystemExit("usage: analyzer.py DATASET.csv")
    rows = list(read_rows(argv[1])[1]); report = spot(rows)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
if __name__ == "__main__": main(sys.argv)
