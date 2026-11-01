#!/usr/bin/env python3
"""NASA93 summary and a grouped categorical effort estimator."""
import argparse, csv, math, random, sys
from collections import Counter

CATEGORICAL = "prec flex resl team pmat rely data cplx ruse docu time stor pvol acap pcap pcon apex plex ltex tool site sced".split()
METADATA = "idX centerX YearX MonthsX".split()
OUTCOMES = {"Kloc+": "maximize", "Effort-": "minimize", "Defects-": "minimize"}
MISSING = {"", "?", "NA", "N/A", "null", "None"}

class Project:
    def __init__(self, position, values):
        self.position, self.values = position, values
    @property
    def identifier(self): return self.values.get("idX", "")

def norm_cell(raw):
    value = "" if raw is None else str(raw).strip()
    return "?" if value in MISSING else value

def parse_float(raw):
    if raw.strip() in MISSING: return None, "missing"
    try:
        value = float(raw.strip())
    except ValueError:
        return None, "invalid"
    return (value, "nonfinite") if not math.isfinite(value) else (value, "valid")

def load(path):
    projects, invalid_rows, issues = [], [], []
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try: header = next(reader)
        except StopIteration: return [], [], [], []
        header = [x.strip() for x in header]
        for position, row in enumerate(reader, 2):
            if not row or all(not x.strip() for x in row): continue
            if len(row) != len(header):
                invalid_rows.append(position); issues.append(f"row {position}: expected {len(header)} fields, got {len(row)}")
                continue
            projects.append(Project(position, dict(zip(header, row))))
    return header, projects, invalid_rows, issues

def summary(path):
    header, projects, invalid_rows, issues = load(path)
    result = {"path": path, "columns": header, "records": len(projects), "invalid_rows": invalid_rows, "issues": issues,
              "roles": {"categorical": CATEGORICAL, "metadata": METADATA, "outcomes": OUTCOMES},
              "categorical": {}, "outcomes": {}, "duplicates": {}}
    for name in CATEGORICAL:
        counts = Counter(p.values.get(name, "") for p in projects if p.values.get(name, "").strip() not in MISSING)
        result["categorical"][name] = dict(sorted(counts.items()))
    for name, goal in OUTCOMES.items():
        valid, missing, invalid, nonfinite = [], 0, 0, 0
        for p in projects:
            value, kind = parse_float(p.values.get(name, ""))
            if kind == "valid": valid.append(value)
            elif kind == "missing": missing += 1
            elif kind == "invalid": invalid += 1
            else: nonfinite += 1
        ordered = sorted(valid); n = len(ordered)
        median = None if not n else (ordered[n//2] if n % 2 else (ordered[n//2-1] + ordered[n//2]) / 2)
        mean = sum(valid) / n if n else None
        variance = sum((x - mean) ** 2 for x in valid) / (n - 1) if n > 1 else 0.0 if n else None
        result["outcomes"][name] = {"goal": goal, "count": n, "missing": missing, "invalid": invalid, "nonfinite": nonfinite,
                                    "min": min(valid) if valid else None, "max": max(valid) if valid else None,
                                    "mean": mean, "median": median, "sample_variance": variance,
                                    "sample_stddev": math.sqrt(variance) if variance is not None else None}
    ids = Counter(p.identifier for p in projects if p.identifier.strip() not in MISSING)
    result["duplicates"] = {k: v for k, v in ids.items() if v > 1}
    return result

def print_summary(data):
    print(f"records: {data['records']}")
    print("columns: " + ", ".join(data["columns"]))
    print("roles: categorical=" + ",".join(data["roles"]["categorical"]) + "; metadata=" + ",".join(data["roles"]["metadata"]) + "; outcomes=Kloc+(maximize), Effort-(minimize), Defects-(minimize)")
    print("CSV positions are 1-based physical lines including the header; blank rows are skipped. Duplicate ids: " + (repr(data["duplicates"]) if data["duplicates"] else "none"))
    for name, counts in data["categorical"].items(): print(f"{name} frequencies: {counts}")
    for name, stats in data["outcomes"].items(): print(f"{name} ({stats['goal']}): {stats}")
    print("missing tokens: " + repr(sorted(MISSING)) + "; malformed rows are skipped and reported; categorical values are never parsed as numeric outcomes.")
    for issue in data["issues"]: print("issue: " + issue)

def _target(project): return parse_float(project.values.get("Effort-", ""))
def descriptor_key(project): return tuple(norm_cell(project.values.get(n, "")) for n in CATEGORICAL)

def split_groups(projects, seed=42, test_fraction=.2):
    groups = {}
    for p in projects: groups.setdefault(descriptor_key(p), []).append(p)
    keys = list(groups); random.Random(seed).shuffle(keys)
    if len(keys) < 2: raise ValueError("at least two descriptor groups are required")
    n_test = min(max(1, round(len(keys) * test_fraction)), len(keys)-1)
    test_keys = set(keys[:n_test])
    return ([p for k in keys if k not in test_keys for p in groups[k]],
            [p for k in keys if k in test_keys for p in groups[k]])

class CategoricalNeighborModel:
    """A symbolic-distance k-neighbor regressor; missing/unseen values are symbols."""
    def __init__(self, neighbors=5): self.neighbors, self.rows, self.mean = neighbors, [], None
    def fit(self, projects):
        self.rows = [(descriptor_key(p), _target(p)[0]) for p in projects if _target(p)[1] == "valid"]
        if not self.rows: raise ValueError("no usable training Effort- targets")
        self.mean = sum(y for _, y in self.rows) / len(self.rows); return self
    def predict_config(self, config):
        if set(config) != set(CATEGORICAL):
            raise ValueError("schema error: missing=%s extra=%s" % (sorted(set(CATEGORICAL)-set(config)), sorted(set(config)-set(CATEGORICAL))))
        key = tuple(norm_cell(config[n]) for n in CATEGORICAL)
        ranked = sorted((sum(a != b for a, b in zip(key, row)), y) for row, y in self.rows)
        chosen = ranked[:min(self.neighbors, len(ranked))]
        return sum(y for _, y in chosen) / len(chosen)
    def predict(self, project): return self.predict_config({n: project.values.get(n, "") for n in CATEGORICAL})

def mae(actual, predicted):
    return None if not actual else sum(abs(a-b) for a, b in zip(actual, predicted)) / len(actual)

def estimate(path, seed=42):
    header, projects, invalid_rows, issues = load(path); train, test = split_groups(projects, seed)
    model = CategoricalNeighborModel().fit(train); actual, predicted, records, excluded = [], [], [], 0
    for p in test:
        y, kind = _target(p)
        if kind != "valid": excluded += 1; continue
        actual.append(y); predicted.append(model.predict(p))
        records.append({"position": p.position, "idX": p.identifier, "actual": y, "predicted": predicted[-1]})
    return {"path": path, "train_rows": len(train), "test_rows": len(test),
            "train_groups": len({descriptor_key(p) for p in train}), "test_groups": len({descriptor_key(p) for p in test}),
            "excluded_test_targets": excluded, "test_mae": mae(actual, predicted),
            "baseline_mae": mae(actual, [model.mean] * len(actual)), "records": records, "model": model,
            "train": train, "test": test, "invalid_rows": invalid_rows, "issues": issues}

def print_estimate(result):
    for key in ("train_rows", "test_rows", "train_groups", "test_groups", "excluded_test_targets", "test_mae", "baseline_mae"):
        print(f"{key}: {result[key]}")
    print("evaluation_records: " + repr(result["records"]))

def _neighbor_evidence(model, query):
    """Return the exact ranked/selected records used by predict_config."""
    key = descriptor_key(query)
    ranked = sorted(((sum(a != b for a, b in zip(key, row[0])), row[1], row[2])
                     for row in model.evidence_rows), key=lambda x: (x[0], x[1]))
    chosen = ranked[:min(model.neighbors, len(ranked))]
    return [{"distance": d, "effort": y, "position": p.position, "idX": p.identifier}
            for d, y, p in chosen]

def _fit_with_evidence(projects, neighbors=5):
    model = CategoricalNeighborModel(neighbors).fit(projects)
    model.evidence_rows = [(descriptor_key(p), _target(p)[0], p)
                           for p in projects if _target(p)[1] == "valid"]
    return model

def marginal_associations(projects):
    """Training-only marginal context; it is not local feature attribution."""
    reports = []
    for name in CATEGORICAL:
        groups, eligible = {}, []
        excluded = Counter()
        for p in projects:
            value = norm_cell(p.values.get(name, "")); effort, kind = _target(p)
            if value == "?": excluded["missing_descriptor"] += 1
            elif kind != "valid": excluded["invalid_or_missing_effort"] += 1
            else: eligible.append((value, effort)); groups.setdefault(value, []).append(effort)
        ys = [y for _, y in eligible]; n = len(ys); ref = sum(ys) / n if n else None
        total = sum((y-ref)**2 for y in ys) if n else None
        between = sum(len(v)*(sum(v)/len(v)-ref)**2 for v in groups.values()) if n else None
        cats = {k: {"count": len(v), "mean_effort": sum(v)/len(v)} for k,v in sorted(groups.items())}
        reports.append({"descriptor": name, "eligible_rows": n, "excluded_rows": sum(excluded.values()),
                        "excluded": dict(excluded), "categories": cats, "reference_mean": ref,
                        "between_ss": between, "total_ss": total,
                        "eta_squared": between/total if total and total > 0 else None})
    return sorted(reports, key=lambda r: (r["eta_squared"] is None, -(r["eta_squared"] or 0), r["descriptor"]))

def justify(path, seed=42, neighbors=5, retrospective=False):
    """Predict the earliest held-out project, explaining it without its outcome."""
    header, projects, invalid_rows, issues = load(path)
    train, test = split_groups(projects, seed)
    query = min(test, key=lambda p: p.position)
    model = _fit_with_evidence(train, neighbors)
    config = {n: query.values.get(n, "") for n in CATEGORICAL}
    prediction = model.predict_config(config)
    evidence = _neighbor_evidence(model, query)
    result = {"path": path, "query": {"position": query.position, "idX": query.identifier,
              "descriptors": dict(config)}, "train_rows": len(train), "test_rows": len(test),
              "train_groups": len({descriptor_key(p) for p in train}),
              "test_groups": len({descriptor_key(p) for p in test}), "predicted_effort": prediction,
              "training_mean_baseline": model.mean, "neighbors": evidence,
              "associations": marginal_associations(train), "invalid_rows": invalid_rows, "issues": issues,
              "retrospective_audit": None}
    if retrospective:
        actual, kind = _target(query)
        result["retrospective_audit"] = {"actual_effort": actual if kind == "valid" else None,
                                         "outcome_kind": kind}
    return result

def print_justify(result):
    print("JUSTIFICATION (prediction and explanation precede any audit)")
    print("query: " + repr(result["query"]))
    print("training_rows: %s test_rows: %s training_groups: %s test_groups: %s" %
          (result["train_rows"], result["test_rows"], result["train_groups"], result["test_groups"]))
    print("predicted_effort: %r" % result["predicted_effort"])
    print("training_mean_baseline: %r" % result["training_mean_baseline"])
    print("neighbors_selected_by_distance_then_effort:")
    for row in result["neighbors"]: print("  " + repr(row))
    print("estimate_reproduction: mean(selected finite training efforts) = %r" %
          (sum(x["effort"] for x in result["neighbors"])/len(result["neighbors"])))
    print("GLOBAL_MARGINAL_ASSOCIATIONS (training-only context; not local contribution or causation):")
    for r in result["associations"]: print("  %s eta_squared=%r categories=%r" % (r["descriptor"], r["eta_squared"], r["categories"]))
    print("assumptions_limitations: observational historical data; missing/unseen descriptors are '?' symbols; invalid training targets are excluded; model uncertainty remains; Effort- units are not invented; at least two descriptor groups and one valid training target are required")
    if result["retrospective_audit"] is not None: print("RETROSPECTIVE_AUDIT (after prediction and explanation): " + repr(result["retrospective_audit"]))

def _betacf(a, b, x):
    tiny, qab, qap, qam = 1e-300, a+b, a+1, a-1
    c, d = 1.0, 1.0-qab*x/qap; d = max(tiny, abs(d)) * (1 if d >= 0 else -1); d = 1/d; h=d
    for m in range(1, 10000):
        m2=2*m; aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; d=max(tiny,abs(d))*(1 if d>=0 else -1); c=1+aa/c; c=max(tiny,abs(c))*(1 if c>=0 else -1); d=1/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; d=max(tiny,abs(d))*(1 if d>=0 else -1); c=1+aa/c; c=max(tiny,abs(c))*(1 if c>=0 else -1); d=1/d; delta=d*c; h*=delta
        if abs(delta-1) < 3e-14: return h
    raise ArithmeticError("incomplete beta continued fraction")

def _ibeta(a, b, x):
    if not 0 <= x <= 1: raise ValueError("beta argument outside [0,1]")
    if x in (0, 1): return x
    bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    return bt*_betacf(a,b,x)/a if x < (a+1)/(a+b+2) else 1-bt*_betacf(b,a,1-x)/b

def t_cdf(t, df):
    if df <= 0 or not math.isfinite(t): raise ValueError("invalid t input")
    if t == 0: return .5
    x=df/(df+t*t); tail=.5*_ibeta(df/2, .5, x)
    return 1-tail if t > 0 else tail

def t_critical_975(df):
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < .975: hi *= 2
    for _ in range(80):
        mid=(lo+hi)/2
        if t_cdf(mid, df) < .975: lo=mid
        else: hi=mid
    return (lo+hi)/2

def certify(projects, threshold=None):
    groups={"n": [], "h": []}; excluded=0
    for p in projects:
        tool=norm_cell(p.values.get("tool", "")).lower(); value, kind=parse_float(p.values.get("Effort-", ""))
        if tool in groups:
            if kind == "valid": groups[tool].append(value)
            else: excluded += 1
    out={"reference":"n", "candidate":"h", "excluded":excluded, "groups":{}}
    for key, vals in groups.items():
        n=len(vals); mean=sum(vals)/n if n else None; var=sum((x-mean)**2 for x in vals)/(n-1) if n>1 else None
        out["groups"][key]={"count":n,"mean":mean,"sample_variance":var}
    if threshold is not None and (not math.isfinite(threshold) or threshold < 0): raise ValueError("threshold must be finite and nonnegative")
    out["threshold"]=threshold; a,b=groups["n"],groups["h"]
    if len(a)<2 or len(b)<2: out["inference"]={"status":"undefined: each group needs at least two usable outcomes"}; return out
    ma,mb=out["groups"]["n"]["mean"],out["groups"]["h"]["mean"]; va,vb=out["groups"]["n"]["sample_variance"],out["groups"]["h"]["sample_variance"]
    diff=mb-ma; se2=va/len(a)+vb/len(b); out["difference"]=diff
    if se2 == 0: out["inference"]={"status":"undefined: zero standard error"}; return out
    se=math.sqrt(se2); df=se2*se2/((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1)); crit=t_critical_975(df); p=2*(1-t_cdf(abs(diff)/se,df)); ci=(diff-crit*se,diff+crit*se)
    out["inference"]={"status":"defined","standard_error":se,"df":df,"t":diff/se,"p_two_sided":max(0,min(1,p)),"ci95":ci}
    if threshold is not None:
        out["practical"]={"point_estimate_meets":diff <= -threshold,"entire_ci_supports":ci[1] <= -threshold}
    return out

def print_certify(result):
    print("comparison: candidate tool=h minus reference tool=n; primary outcome=Effort- (lower is better)")
    for k,v in result["groups"].items(): print(f"{k}: {v}")
    print("difference: " + repr(result.get("difference"))); print("inference: " + repr(result["inference"]))
    print("threshold: " + repr(result.get("threshold"))); print("practical: " + repr(result.get("practical")))

def flow(path, neighbors=5):
    """Replay rows in CSV order, predicting before revealing each effort."""
    header, projects, invalid_rows, issues = load(path)
    history = []
    records = []
    actual, model_errors, baseline_errors = [], [], []
    warmup = invalid = 0
    for project in projects:
        prior_mean = (sum(y for _, y in history) / len(history)) if history else None
        prediction = None
        if history:
            model = CategoricalNeighborModel(neighbors)
            model.rows = list(history); model.mean = prior_mean
            prediction = model.predict(project)
        else:
            warmup += 1
        effort, kind = _target(project)
        if kind != "valid":
            invalid += 1
        evaluated = prediction is not None and kind == "valid"
        if evaluated:
            actual.append(effort); model_errors.append(abs(effort - prediction))
            baseline_errors.append(abs(effort - prior_mean))
        record = {"position": project.position, "idX": project.identifier,
                  "prediction": prediction, "baseline_prediction": prior_mean,
                  "actual_revealed": effort if kind == "valid" else None,
                  "outcome_kind": kind, "evaluated": evaluated,
                  "history_count_before": len(history)}
        records.append(record)
        if kind == "valid": history.append((descriptor_key(project), effort))
    return {"path": path, "records": records, "warmup_count": warmup,
            "evaluated_count": len(actual), "invalid_outcome_count": invalid,
            "final_history_count": len(history), "model_mae": sum(model_errors)/len(model_errors) if model_errors else None,
            "baseline_mae": sum(baseline_errors)/len(baseline_errors) if baseline_errors else None,
            "invalid_rows": invalid_rows, "issues": issues}

def print_flow(result):
    for key in ("warmup_count", "evaluated_count", "invalid_outcome_count", "final_history_count", "model_mae", "baseline_mae"):
        print(f"{key}: {result[key]}")
    print("arrival_records: " + repr(result["records"]))

def choose(path, cap=100):
    """Suggest the smallest observed training-config change for a held-out query."""
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not math.isfinite(cap) or cap < 0:
        raise ValueError("cap must be finite and nonnegative")
    header, projects, invalid_rows, issues = load(path)
    if not projects: raise ValueError("no parsed project rows")
    train, test = split_groups(projects, 42)
    query = min(test, key=lambda p: p.position)
    old = {n: norm_cell(query.values.get(n, "")) for n in CATEGORICAL}
    if any(v == "?" for v in old.values()):
        return {"status": "invalid/incomplete baseline", "reason": "baseline has missing descriptors", "cap": cap, "query": {"position": query.position, "idX": query.identifier, "descriptors": old}}
    model = _fit_with_evidence(train)
    baseline = model.predict_config(old)
    valid_training = [p for p in train if _target(p)[1] == "valid" and not any(norm_cell(p.values.get(n, "")) == "?" for n in CATEGORICAL)]
    candidates, excluded = [], 0
    for p in train:
        config = {n: norm_cell(p.values.get(n, "")) for n in CATEGORICAL}
        if any(v == "?" for v in config.values()) or _target(p)[1] != "valid": excluded += 1; continue
        prediction = model.predict_config(config)
        changes = [n for n in CATEGORICAL if old[n] != config[n]]
        if prediction <= cap and prediction < baseline: candidates.append((len(changes), prediction, p.position, p, config, changes))
    base_report = {"position": query.position, "idX": query.identifier, "descriptors": old, "predicted_effort": baseline}
    common = {"path": path, "cap": cap, "baseline": base_report, "training_rows": len(train), "test_rows": len(test), "candidate_count": len(candidates), "excluded_candidate_count": excluded, "invalid_rows": invalid_rows, "issues": issues}
    if baseline <= cap: common.update(status="already feasible", recommendation="no change")
    elif not valid_training: common.update(status="insufficient training data", recommendation=None)
    elif not candidates: common.update(status="no feasible improving candidate", recommendation=None)
    else:
        _, prediction, _, winner, config, changes = min(candidates, key=lambda x: (x[0], x[1], x[2]))
        common.update(status="successful recommendation", recommendation={"position": winner.position, "idX": winner.identifier, "descriptors": config, "old_values": {n: old[n] for n in changes}, "new_values": {n: config[n] for n in changes}, "changed_fields": len(changes), "predicted_effort": prediction, "predicted_improvement": baseline-prediction})
    common["interpretation"] = "model-based historical suggestion; not a causal or actionable prescription; feasibility does not guarantee actual effort or preserve size, quality, or other requirements"
    return common

def print_choose(result):
    print("BUDGET_FIX_RECOMMENDATION: " + repr(result))
    print("interpretation: " + result["interpretation"])

def blame(path):
    """Rank marginal categorical associations with valid Effort- values."""
    _, projects, invalid_rows, issues = load(path)
    reports = []
    for name in CATEGORICAL:
        groups = {}
        excluded = Counter()
        eligible_rows = []
        for p in projects:
            raw = p.values.get(name, "")
            value = norm_cell(raw)
            effort, kind = _target(p)
            if value == "?":
                excluded["missing_descriptor"] += 1
            elif kind != "valid":
                excluded["invalid_or_missing_effort"] += 1
            else:
                eligible_rows.append((value, effort))
                groups.setdefault(value, []).append(effort)
        ys = [y for _, y in eligible_rows]
        n = len(ys)
        reference = sum(ys) / n if n else None
        total = sum((y - reference) ** 2 for y in ys) if n else None
        between = sum(len(v) * (sum(v) / len(v) - reference) ** 2 for v in groups.values()) if n else None
        score = (between / total) if total and total > 0 else None
        categories = {}
        for category in sorted(groups):
            values = groups[category]
            categories[category] = {"count": len(values), "mean_effort": sum(values) / len(values)}
        if reference is None:
            higher, lower = [], []
        else:
            higher = [k for k, v in categories.items() if v["mean_effort"] > reference]
            lower = [k for k, v in categories.items() if v["mean_effort"] < reference]
        reports.append({"descriptor": name, "eligible_rows": n,
                        "excluded_rows": sum(excluded.values()), "excluded": dict(excluded),
                        "categories": categories, "reference_mean": reference,
                        "between_ss": between, "total_ss": total, "eta_squared": score,
                        "higher_mean_effort": higher, "lower_mean_effort": lower})
    ranked = sorted(reports, key=lambda r: (r["eta_squared"] is None,
                                             -(r["eta_squared"] or 0), r["descriptor"]))
    for rank, report in enumerate(ranked, 1): report["rank"] = rank
    return {"path": path, "reports": ranked, "invalid_rows": invalid_rows, "issues": issues,
            "score": "eta_squared = between-group SS / total SS; undefined when no eligible rows or total SS is zero",
            "interpretation": "Marginal historical associations; lower Effort- is better. They are not causal or independent effects."}

def print_blame(result):
    print("score: " + result["score"])
    print(result["interpretation"])
    print("ranked_associations:")
    for r in result["reports"]:
        print(f"rank={r['rank']} descriptor={r['descriptor']} eligible={r['eligible_rows']} excluded={r['excluded_rows']} "
              f"reference_mean={r['reference_mean']} between_ss={r['between_ss']} total_ss={r['total_ss']} eta_squared={r['eta_squared']}")
        print("  excluded=" + repr(r["excluded"]) + " categories=" + repr(r["categories"]))
        print("  higher_mean_effort=" + repr(r["higher_mean_effort"]) + " lower_mean_effort=" + repr(r["lower_mean_effort"]))

def _median(values):
    ordered = sorted(values); n = len(ordered)
    return None if not n else ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2

def spot(path, threshold=3.5, min_reference=3):
    """Leave-one-out robust screening of log1p(Effort-), marginally."""
    if not math.isfinite(threshold) or threshold < 0: raise ValueError("threshold must be finite and nonnegative")
    if not isinstance(min_reference, int) or min_reference < 0: raise ValueError("min_reference must be a nonnegative integer")
    _, projects, invalid_rows, issues = load(path); valid = []
    for p in projects:
        effort, kind = _target(p)
        if kind == "valid" and effort >= 0: valid.append((p, effort, math.log1p(effort)))
    reports = []
    for p in projects:
        effort, kind = _target(p); base = {"position": p.position, "idX": p.identifier,
            "observed_effort": effort if kind in ("valid", "nonfinite") else None,
            "transformed": None, "reference_count": 0, "center": None, "scale": None,
            "score": None, "threshold": threshold, "descriptor": {n: norm_cell(p.values.get(n, "")) for n in CATEGORICAL}}
        if kind != "valid": base.update(status="invalid_or_missing", reason="missing, invalid, or nonfinite Effort- outcome")
        elif effort < 0: base.update(status="invalid_or_missing", reason="negative Effort- cannot be transformed by log1p")
        else:
            transformed = math.log1p(effort); refs = [x[2] for x in valid if x[0] is not p]
            base["transformed"] = transformed; base["reference_count"] = len(refs)
            if len(refs) < min_reference: base.update(status="undefined", reason="insufficient leave-one-out references")
            else:
                center = _median(refs); deviations = [abs(x-center) for x in refs]; scale = _median(deviations)
                base.update(center=center, scale=scale)
                if scale == 0: base.update(status="undefined", reason="zero MAD; score is not defined")
                else:
                    score = abs(transformed-center) / scale; base["score"] = score
                    base.update(status="flagged" if score >= threshold else "ordinary_nonflag",
                                reason="robust score meets threshold" if score >= threshold else "valid observation below threshold")
        reports.append(base)
    counts = Counter(r["status"] for r in reports)
    return {"path": path, "threshold": threshold, "min_reference": min_reference, "reports": reports,
            "counts": {"total_valid": sum(r["status"] != "invalid_or_missing" for r in reports),
                       "excluded": counts["invalid_or_missing"], "flagged": counts["flagged"],
                       "ordinary_nonflag": counts["ordinary_nonflag"], "undefined": counts["undefined"]},
            "invalid_rows": invalid_rows, "issues": issues,
            "policy": "log1p nonnegative Effort-; leave-one-out median/MAD; zero MAD and short references undefined",
            "interpretation": "Marginal historical effort screening; size and other descriptors are not adjusted. Unusual results warrant investigation, not proof of measurement error; both low and high effort may be unusual."}

def print_spot(result, examples=12):
    print("policy: " + result["policy"]); print(result["interpretation"])
    print("counts: " + repr(result["counts"])); print("positions are 1-based physical CSV lines including header; malformed rows are skipped and reported")
    for row in result["reports"]:
        if row["status"] != "ordinary_nonflag": print("spot: " + repr(row))

def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize or estimate NASA93 historical projects")
    parser.add_argument("dataset")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--flow", action="store_true")
    parser.add_argument("--blame", action="store_true")
    parser.add_argument("--justify", action="store_true")
    parser.add_argument("--retrospective-audit", action="store_true")
    parser.add_argument("--choose", action="store_true")
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--spot-threshold", type=float, default=3.5)
    parser.add_argument("--spot-min-reference", type=int, default=3)
    parser.add_argument("--cap", type=float, default=100)
    args = parser.parse_args(argv)
    if args.spot: print_spot(spot(args.dataset, args.spot_threshold, args.spot_min_reference))
    elif args.choose: print_choose(choose(args.dataset, args.cap))
    elif args.justify: print_justify(justify(args.dataset, retrospective=args.retrospective_audit))
    elif args.blame: print_blame(blame(args.dataset))
    elif args.flow: print_flow(flow(args.dataset))
    elif args.certify:
        _, rows, _, _ = load(args.dataset); ref=[parse_float(p.values.get("Effort-", ""))[0] for p in rows if norm_cell(p.values.get("tool", "")).lower()=="n" and parse_float(p.values.get("Effort-", ""))[1]=="valid"]
        threshold=0.10*(sum(ref)/len(ref)) if ref else None; print_certify(certify(rows, threshold))
    elif args.estimate: print_estimate(estimate(args.dataset))
    else: print_summary(summary(args.dataset))
    return 0

if __name__ == "__main__": sys.exit(main())
