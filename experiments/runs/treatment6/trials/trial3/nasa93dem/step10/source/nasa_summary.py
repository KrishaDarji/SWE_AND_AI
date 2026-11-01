"""NASA93 historical-project loader and descriptive summary (stdlib only)."""
import argparse, csv, math, pathlib, statistics, sys, copy
import random

CATEGORICAL = {"prec","flex","resl","team","pmat","rely","data","cplx","ruse","docu","time","stor","pvol","acap","pcap","pcon","apex","plex","ltex","tool","site","sced"}
METADATA = {"idX","centerX","YearX","MonthsX"}
OUTCOMES = {"Kloc+":"maximize", "Effort-":"minimize", "Defects-":"minimize"}
MISSING = {"", "?", "NA", "N/A", "null", "NULL", "None"}

PREDICTORS = ("prec","flex","resl","team","pmat","rely","data","cplx","ruse","docu","time","stor","pvol","acap","pcap","pcon","apex","plex","ltex","tool","site","sced")

def _cell(value):
    value = "" if value is None else str(value).strip()
    return "?" if value in MISSING else value

def _effort(value):
    try:
        x = float(str(value).strip())
        return x if math.isfinite(x) else None
    except (AttributeError, ValueError):
        return None

def _groups(projects):
    out = {}
    for p in projects:
        key = tuple(_cell(p["values"].get(c, "")) for c in PREDICTORS)
        out.setdefault(key, []).append(p)
    return out

def split_groups(projects, seed=42, fraction=.8):
    groups = list(_groups(projects).values())
    if len(groups) < 2: raise ValueError("at least two descriptor groups are required")
    random.Random(seed).shuffle(groups)
    cut = max(1, min(len(groups)-1, round(len(groups)*fraction)))
    return [p for g in groups[:cut] for p in g], [p for g in groups[cut:] for p in g]

class CategoricalKNN:
    """EZR-inspired symbolic distance; nearest training row, mean tie prediction."""
    def __init__(self, rows):
        self.rows = []
        self.training_mean = None
        ys = []
        for p in rows:
            y = _effort(p["values"].get("Effort-", ""))
            if y is not None:
                ys.append(y); self.rows.append((tuple(_cell(p["values"].get(c,"")) for c in PREDICTORS), y, p))
        if ys: self.training_mean = statistics.fmean(ys)
    def predict(self, config):
        if self.training_mean is None: raise ValueError("no usable training targets")
        key = _validate_config(config)
        if not self.rows: return self.training_mean
        distances = [sum(a != b for a,b in zip(key, x[0])) for x in self.rows]
        best = min(distances)
        return statistics.fmean(x[1] for x,d in zip(self.rows, distances) if d == best)

    def evidence(self, config):
        key = _validate_config(config)
        if not self.rows: raise ValueError("no usable training targets")
        distances = [sum(a != b for a,b in zip(key, x[0])) for x in self.rows]
        best = min(distances)
        return [{"position": p.get("position"), "idX": p["values"].get("idX", ""), "distance": d, "effort": y}
                for (x,y,p), d in zip(self.rows, distances) if d == best]

def _validate_config(config):
    if not isinstance(config, dict): raise TypeError("configuration must be a mapping")
    missing = [c for c in PREDICTORS if c not in config]
    extra = sorted(set(config) - set(PREDICTORS))
    if missing or extra: raise ValueError(f"schema error: missing={missing}, extra={extra}")
    return tuple(_cell(config[c]) for c in PREDICTORS)

def evaluate(rows, seed=42):
    train, test = split_groups(rows, seed)
    model = CategoricalKNN(train)
    scored = [(p, _effort(p["values"].get("Effort-", ""))) for p in test]
    scored = [(p,y,model.predict({c: p["values"].get(c, "") for c in PREDICTORS})) for p,y in scored if y is not None]
    baseline = model.training_mean
    mae = (statistics.fmean(abs(y-pred) for _,y,pred in scored) if scored else None)
    bmae = (statistics.fmean(abs(y-baseline) for _,y,_ in scored) if scored and baseline is not None else None)
    return {"model": model, "train": train, "test": test, "scored": scored, "mae": mae, "baseline_mae": bmae,
            "train_groups": len(_groups(train)), "test_groups": len(_groups(test)),
            "excluded_targets": sum(_effort(p["values"].get("Effort-", "")) is None for p in rows)}

def justify(rows, seed=42):
    train, test = split_groups(rows, seed)
    if not train or not test: raise ValueError("both training and held-out partitions are required")
    query = min(test, key=lambda p: p.get("position", 0))
    model = CategoricalKNN(train)
    config = {c: query["values"].get(c, "") for c in PREDICTORS}
    return {"query": query, "train": train, "test": test, "prediction": model.predict(config),
            "training_mean": model.training_mean, "neighbors": model.evidence(config),
            "associations": blame({"projects": train}), "train_groups": len(_groups(train)),
            "test_groups": len(_groups(test)), "actual_audit": _effort(query["values"].get("Effort-", ""))}

def render_justify(path):
    result = justify(load(path)["projects"]); q = result["query"]; v = q["values"]
    print("NASA93 estimate justification (prediction completed before retrospective audit)")
    print(f"query_position={q.get('position')} query_idX={v.get('idX','')} train_rows={len(result['train'])} test_rows={len(result['test'])} train_groups={result['train_groups']} test_groups={result['test_groups']}")
    print("query_descriptors=" + str({c: v.get(c, "") for c in PREDICTORS}))
    print(f"predicted_effort={result['prediction']} training_mean_baseline={result['training_mean']}")
    print("neighbors=" + str(result["neighbors"]))
    print("neighbor_rule=Hamming distance over the exact 22 categorical descriptors; all tied nearest finite-target training rows are averaged.")
    print("global_associations_training_only=" + str(result["associations"]))
    print("association_label=global marginal training association, not local contribution, causation, or guaranteed benefit")
    print(f"retrospective_audit_actual_effort={result['actual_audit']}")
    print("assumptions=observational historical data; missing/unseen descriptors are literal categories; invalid training targets are excluded; no effort units are invented.")
    print("limitations=model uncertainty, insufficient groups, and unavailable estimates are explicit errors; duplicate idX values remain identified by original position.")

def render_estimate(path):
    result = evaluate(load(path)["projects"])
    print(f"train_rows={len(result['train'])} test_rows={len(result['test'])} train_groups={result['train_groups']} test_groups={result['test_groups']}")
    print(f"excluded_targets={result['excluded_targets']} heldout_mae={result['mae']} training_mean_baseline_mae={result['baseline_mae']}")

def flow(rows):
    """Replay rows in input order; predictions see only earlier valid outcomes."""
    history, records, model_errors, baseline_errors = [], [], [], []
    for arrival, project in enumerate(rows, 1):
        values = project["values"]; config = {c: values.get(c, "") for c in PREDICTORS}
        model = CategoricalKNN(history) if history else None
        prediction = model.predict(config) if model else None
        baseline = model.training_mean if model else None
        actual = _effort(values.get("Effort-", "")); evaluated = actual is not None and prediction is not None
        record = {"arrival": arrival, "position": project.get("position"), "idX": values.get("idX", ""),
                  "prediction": prediction, "baseline_prediction": baseline, "actual_revealed": actual,
                  "evaluated": evaluated, "model_error": abs(actual-prediction) if evaluated else None,
                  "baseline_error": abs(actual-baseline) if evaluated else None, "history_count": len(history)}
        records.append(copy.deepcopy(record))
        if evaluated: model_errors.append(record["model_error"]); baseline_errors.append(record["baseline_error"])
        if actual is not None: history.append(copy.deepcopy(project))
    return {"records": records, "warm_up_count": sum(r["prediction"] is None for r in records),
            "evaluated_count": len(model_errors), "invalid_outcome_count": sum(_effort(p["values"].get("Effort-", "")) is None for p in rows),
            "mae": statistics.fmean(model_errors) if model_errors else None,
            "baseline_mae": statistics.fmean(baseline_errors) if baseline_errors else None,
            "model_errors": model_errors, "baseline_errors": baseline_errors}

def render_flow(path):
    result = flow(load(path)["projects"])
    print("NASA93 arrival flow")
    print(f"warm_up={result['warm_up_count']} evaluated={result['evaluated_count']} invalid_outcomes={result['invalid_outcome_count']}")
    print(f"model_mae={result['mae']} baseline_mae={result['baseline_mae']}")
    for record in result["records"][:3] + result["records"][-3:]: print("arrival_record=" + str(record))

def _number(value):
    try:
        x = float(value.strip())
        return x if math.isfinite(x) else None
    except (AttributeError, ValueError):
        return None

def median(values):
    """Median using the conventional middle/two-middle average."""
    xs = sorted(values)
    if not xs: return None
    n = len(xs); mid = n // 2
    return xs[mid] if n % 2 else (xs[mid-1] + xs[mid]) / 2

def numeric_summary(values):
    xs = [x for x in values if x is not None]
    if not xs: return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "sample_variance": None}
    mean = statistics.fmean(xs)
    variance = statistics.variance(xs) if len(xs) > 1 else None
    return {"count": len(xs), "min": min(xs), "max": max(xs), "mean": mean,
            "median": median(xs), "sample_variance": variance}

def load(path):
    """Load 1-based CSV data positions; blank lines are ignored, malformed rows retained as errors."""
    result = {"path": str(path), "columns": [], "projects": [], "errors": [], "blank_rows": 0}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try: header = next(reader)
        except StopIteration: return result
        result["columns"] = [x.strip() for x in header]
        width = len(header)
        for position, row in enumerate(reader, 2):
            if not row or all(not cell.strip() for cell in row):
                result["blank_rows"] += 1; continue
            if len(row) != width:
                result["errors"].append({"position": position, "kind": "malformed_row", "detail": f"expected {width} fields, got {len(row)}"}); continue
            cells = {name: value.strip() for name, value in zip(result["columns"], row)}
            result["projects"].append({"position": position, "values": cells})
    return result

def summarize(data):
    cols = data["columns"]
    roles = {c: ("categorical" if c in CATEGORICAL else "metadata" if c in METADATA else "outcome" if c in OUTCOMES else "unclassified") for c in cols}
    missing, invalid, categorical = {}, {}, {}
    for col in cols:
        vals = [p["values"].get(col, "") for p in data["projects"]]
        missing[col] = sum(v in MISSING for v in vals)
        if col in CATEGORICAL:
            categorical[col] = {}
            for v in vals:
                if v not in MISSING: categorical[col][v] = categorical[col].get(v, 0) + 1
        elif col in OUTCOMES:
            invalid[col] = sum(v not in MISSING and _number(v) is None for v in vals)
    outcomes = {c: numeric_summary([_number(p["values"].get(c, "")) for p in data["projects"]]) for c in OUTCOMES if c in cols}
    ids = [p["values"].get("idX", "") for p in data["projects"]]
    duplicates = sorted({x for x in ids if x and ids.count(x) > 1})
    return {"record_count": len(data["projects"]), "columns": cols, "roles": roles, "missing": missing,
            "invalid_numeric": invalid, "categorical_frequencies": categorical, "outcomes": outcomes,
            "duplicate_idX": duplicates, "malformed_rows": data["errors"], "blank_rows": data["blank_rows"]}

def render(data):
    s = summarize(data); print("NASA93 historical project summary")
    print(f"records={s['record_count']} columns={','.join(s['columns'])}")
    print("roles=" + "; ".join(f"{k}:{v}" for k,v in s["roles"].items()))
    print("missing=" + str(s["missing"]))
    print("invalid_numeric=" + str(s["invalid_numeric"]))
    print("categorical_frequencies=" + str(s["categorical_frequencies"]))
    print("outcomes=" + str(s["outcomes"]))
    print("duplicate_idX=" + str(s["duplicate_idX"]))
    print("malformed_rows=" + str(s["malformed_rows"]))
    print("convention=Kloc+ is maximized; Effort- and Defects- are minimized; this is a dataset goal convention, not a claim that larger projects are inherently better.")
    print("positions=CSV physical line numbers, header is line 1; blank data rows are ignored; malformed nonblank rows are excluded and reported.")
    print("assumptions=comma-separated CSV with one header; whitespace is stripped; missing tokens are '', ?, NA, N/A, null, NULL, None; categorical tokens are never parsed as numbers.")
    print("limitations=only declared outcomes receive numeric validation; unknown columns are unclassified; no units or metric formulas are inferred.")

def _betacf(a,b,x):
    qab,qap,qam=a+b,a+1,a-1; c=1.; d=1.-qab*x/qap; d=max(abs(d),1e-300)*(1 if d>=0 else -1); h=d
    for m in range(1,201):
        m2=2*m; aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; d=max(abs(d),1e-300)*(1 if d>=0 else -1); c=1+aa/c; c=max(abs(c),1e-300)*(1 if c>=0 else -1); d=1/d; h*=d*c; aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; d=max(abs(d),1e-300)*(1 if d>=0 else -1); c=1+aa/c; c=max(abs(c),1e-300)*(1 if c>=0 else -1); d=1/d; dl=d*c; h*=dl
        if abs(dl-1)<3e-14: break
    return h
def _ibeta(x,a,b):
    if x in (0,1): return x
    f=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x)); return f*_betacf(a,b,x)/a if x < (a+1)/(a+b+2) else 1-f*_betacf(b,a,1-x)/b
def t_cdf(t,df):
    if not math.isfinite(t) or df<=0:return None
    if t==0:return .5
    z=abs(t); n=max(2000,int(z*1000)); n += n%2
    scale=math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi)
    h=z/n; total=0.
    for i in range(n+1):
        q=i*h; w=1 if i in (0,n) else (4 if i%2 else 2)
        total += w*scale*(1+q*q/df)**(-(df+1)/2)
    area=total*h/3; return .5+area if t>0 else .5-area
def welch_test(reference,candidate,threshold=None):
    r=[float(x) for x in reference if x is not None and math.isfinite(float(x))]; c=[float(x) for x in candidate if x is not None and math.isfinite(float(x))]; out={"reference":numeric_summary(r),"candidate":numeric_summary(c)}
    if len(r)<2 or len(c)<2: out["status"]="insufficient samples"; return out
    vr,vc=statistics.variance(r),statistics.variance(c); diff=statistics.fmean(c)-statistics.fmean(r); se=math.sqrt(vr/len(r)+vc/len(c)); out.update(difference=diff,standard_error=se)
    if se==0: out["status"]="zero standard error"; return out
    df=(vr/len(r)+vc/len(c))**2/((vr/len(r))**2/(len(r)-1)+(vc/len(c))**2/(len(c)-1)); t=diff/se; lo,hi=0.,20.
    while t_cdf(hi,df)<.975: hi*=2
    for _ in range(80):
        mid=(lo+hi)/2
        if t_cdf(mid,df)<.975: lo=mid
        else: hi=mid
    crit=(lo+hi)/2; p=2*min(t_cdf(t,df),1-t_cdf(t,df)); out.update(df=df,t=t,p_two_sided=max(0.,min(1.,p)),ci95=(diff-crit*se,diff+crit*se))
    if threshold is not None:
        if not math.isfinite(threshold) or threshold<0: raise ValueError("threshold must be finite and nonnegative")
        out["threshold"]={"value":threshold,"point_meets":diff<=-threshold,"interval_supports":out["ci95"][1]<=-threshold}
    return out
def certify(path):
    r=[]; c=[]
    for p in load(path)["projects"]:
        v=p["values"]; y=_effort(v.get("Effort-","")); tool=_cell(v.get("tool",""))
        if y is not None and tool in ("n","h"): (r if tool=="n" else c).append(y)
    threshold=.1*statistics.fmean(r) if r else None; result=welch_test(r,c,threshold) if threshold is not None else welch_test(r,c)
    print("certify comparison: reference tool=n, candidate tool=h, outcome=Effort- (lower is better)"); print(f"reference={result['reference']} candidate={result['candidate']}"); print(f"difference(candidate-reference)={result.get('difference')} test={result.get('p_two_sided')} df={result.get('df')} ci95={result.get('ci95')} threshold_formula=10% of reference mean Effort- threshold={threshold}"); print("interpretation=statistical evidence, point-estimate improvement, and whole-interval support are separate; nonsignificance is not equivalence."); print("caveat=historical projects differ in size and other characteristics; this comparison does not establish causal tool effects, productivity improvement, or future performance; effort units are not invented.")

def blame(data):
    """Rank marginal categorical associations with Effort- using eta-squared."""
    rows = data["projects"] if isinstance(data, dict) else list(data)
    results = []
    for descriptor in PREDICTORS:
        eligible, excluded = [], 0
        for project in rows:
            value = _cell(project["values"].get(descriptor, ""))
            effort = _effort(project["values"].get("Effort-", ""))
            if value == "?" or effort is None: excluded += 1
            else: eligible.append((value, effort))
        groups = {}
        for value, effort in eligible: groups.setdefault(value, []).append(effort)
        mean = statistics.fmean(y for _, y in eligible) if eligible else None
        categories, between = [], 0.0
        for value in sorted(groups):
            ys = groups[value]; group_mean = statistics.fmean(ys)
            between += len(ys) * (group_mean - mean) ** 2
            categories.append({"value": value, "count": len(ys), "mean_effort": group_mean})
        total = sum((y - mean) ** 2 for _, y in eligible) if eligible else None
        score = between / total if total and total > 0 else None
        direction = {"higher_mean": [x["value"] for x in categories if mean is not None and x["mean_effort"] > mean],
                     "lower_mean": [x["value"] for x in categories if mean is not None and x["mean_effort"] < mean]}
        results.append({"descriptor": descriptor, "eligible_count": len(eligible), "excluded_count": excluded,
                        "categories": categories, "reference_mean": mean, "between_ss": between if eligible else None,
                        "total_ss": total, "eta_squared": score, "direction": direction})
    ranked = sorted(results, key=lambda x: (x["eta_squared"] is None, -(x["eta_squared"] or 0), x["descriptor"]))
    for rank, result in enumerate(ranked, 1): result["rank"] = rank
    return ranked

def render_blame(data):
    ranked = blame(data)
    print("NASA93 marginal blame: Effort- (lower observed mean is better)")
    print("score=eta_squared=between_group_sum_of_squares/total_sum_of_squares; undefined when no eligible rows or total SS is zero")
    print("interpretation=marginal historical association, not causation, independent importance, significance, productivity effect, or guaranteed improvement; size, correlation, and unequal category sizes can affect ranking; effort units are not invented")
    for item in ranked:
        cats = ";".join(f"{x['value']}:{x['count']}:{x['mean_effort']:.6g}" for x in item["categories"])
        print(f"rank={item['rank']} descriptor={item['descriptor']} eligible={item['eligible_count']} excluded={item['excluded_count']} reference_mean={item['reference_mean']} between_ss={item['between_ss']} total_ss={item['total_ss']} eta_squared={item['eta_squared']} higher={item['direction']['higher_mean']} lower={item['direction']['lower_mean']} categories={cats}")

def choose(data, effort_cap=100):
    """Retrospectively select the largest observed project within an effort cap."""
    if not isinstance(effort_cap, (int, float)) or isinstance(effort_cap, bool) or not math.isfinite(effort_cap) or effort_cap < 0:
        raise ValueError("effort_cap must be finite and nonnegative")
    projects = data.get("projects", []) if isinstance(data, dict) else list(data)
    valid, feasible, invalid, over_budget = [], [], [], []
    for project in projects:
        values = project.get("values", {})
        descriptors_ok = all(_cell(values.get(c, "")) != "?" for c in PREDICTORS)
        kloc = _number(values.get("Kloc+", "")); effort = _effort(values.get("Effort-", "")); defects = _number(values.get("Defects-", ""))
        if not descriptors_ok or kloc is None or effort is None or defects is None or effort < 0:
            invalid.append(project); continue
        valid.append(project)
        (feasible if effort <= effort_cap else over_budget).append(project)
    winner = min(feasible, key=lambda p: (-_number(p["values"]["Kloc+"]), _effort(p["values"]["Effort-"]), p.get("position", 0))) if feasible else None
    return {"winner": winner, "cap": effort_cap, "raw_count": len(projects), "valid_count": len(valid), "feasible_count": len(feasible), "invalid_data_count": len(invalid), "over_budget_count": len(over_budget), "malformed_row_count": len(data.get("errors", [])) if isinstance(data, dict) else 0, "selection_rule": "highest Kloc+, then lowest Effort-, then earliest original CSV position; Defects- descriptive only"}

def render_choose(path, effort_cap=100):
    data = load(path); result = choose(data, effort_cap); winner = result["winner"]
    print("NASA93 retrospective selection among historical projects")
    print(f"raw_parsed_count={result['raw_count']} valid_count={result['valid_count']} feasible_count={result['feasible_count']}")
    print(f"invalid_data_exclusion_count={result['invalid_data_count']} over_budget_exclusion_count={result['over_budget_count']} malformed_row_count={result['malformed_row_count']}")
    print(f"cap={result['cap']} feasibility=complete descriptors and finite nonnegative Effort-, Kloc+, Defects-, with Effort- <= cap")
    print(f"selection_rule={result['selection_rule']}")
    if winner is None: print("winner=None")
    else:
        values = winner["values"]
        print(f"winner_position={winner.get('position')} winner_idX={values.get('idX', '')}")
        print("winner_descriptors=" + str({c: _cell(values.get(c, "")) for c in PREDICTORS}))
        print(f"winner_Kloc+={_number(values.get('Kloc+', ''))} winner_Effort-={_effort(values.get('Effort-', ''))} winner_Defects-={_number(values.get('Defects-', ''))}")
    print("goal_note=Maximizing Kloc+ is this task's convention; it does not establish productivity, causation, suitability, or a guaranteed future budget.")

def fix(rows, effort_cap=100, seed=42):
    """Suggest the smallest training-observed descriptor change under a model cap."""
    if not isinstance(effort_cap, (int, float)) or isinstance(effort_cap, bool) or not math.isfinite(effort_cap) or effort_cap < 0:
        raise ValueError("effort_cap must be finite and nonnegative")
    train, test = split_groups(rows, seed)
    if not train or not test:
        return {"status": "insufficient training data", "cap": effort_cap, "train": train, "test": test}
    query = min(test, key=lambda p: p.get("position", 0)); values = query.get("values", {})
    baseline_config = {c: values.get(c, "") for c in PREDICTORS}
    try: baseline_key = _validate_config(baseline_config)
    except (TypeError, ValueError) as exc:
        return {"status": "invalid/incomplete baseline", "cap": effort_cap, "query": query, "error": str(exc), "train": train, "test": test}
    if "?" in baseline_key:
        return {"status": "invalid/incomplete baseline", "cap": effort_cap, "query": query, "error": "baseline has missing descriptors", "train": train, "test": test}
    model = CategoricalKNN(train)
    if model.training_mean is None:
        return {"status": "insufficient training data", "cap": effort_cap, "query": query, "train": train, "test": test}
    baseline_prediction = model.predict(baseline_config)
    result = {"status": "already feasible" if baseline_prediction <= effort_cap else "no feasible improving candidate", "cap": effort_cap, "query": query, "train": train, "test": test,
              "baseline": {"descriptors": dict(zip(PREDICTORS, baseline_key)), "position": query.get("position"), "idX": values.get("idX", ""), "prediction": baseline_prediction}, "candidate_count": 0, "excluded_candidate_count": 0}
    if baseline_prediction <= effort_cap:
        result["recommendation"] = None; return result
    seen, candidates = set(), []
    for project in train:
        v = project.get("values", {}); config = {c: v.get(c, "") for c in PREDICTORS}
        try: key = _validate_config(config)
        except (TypeError, ValueError): result["excluded_candidate_count"] += 1; continue
        if "?" in key or key in seen: result["excluded_candidate_count"] += 1; continue
        seen.add(key); result["candidate_count"] += 1; prediction = model.predict(config)
        changed = [c for c, old, new in zip(PREDICTORS, baseline_key, key) if old != new]
        if prediction <= effort_cap and prediction < baseline_prediction:
            candidates.append({"descriptors": dict(zip(PREDICTORS, key)), "old_values": {c: old for c, old, new in zip(PREDICTORS, baseline_key, key) if old != new}, "new_values": {c: new for c, old, new in zip(PREDICTORS, baseline_key, key) if old != new}, "changed_field_count": len(changed), "position": project.get("position"), "idX": v.get("idX", ""), "prediction": prediction, "predicted_improvement": baseline_prediction - prediction})
    candidates.sort(key=lambda x: (x["changed_field_count"], x["prediction"], x["position"]))
    result["candidates"] = candidates; result["recommendation"] = candidates[0] if candidates else None
    if candidates: result["status"] = "successful recommendation"
    result["audit"] = {"query_actual_effort": _effort(values.get("Effort-", "")), "label": "retrospective audit; excluded from selection and prediction"}
    return result

def render_fix(path, effort_cap=100):
    result = fix(load(path)["projects"], effort_cap)
    print("NASA93 model-based budget fix (training configurations only)")
    print("status=" + result["status"] + " cap=" + str(result["cap"]))
    if "baseline" in result: print("baseline=" + str(result["baseline"]))
    print(f"candidate_count={result.get('candidate_count', 0)} excluded_candidate_count={result.get('excluded_candidate_count', 0)}")
    print("recommendation=" + str(result.get("recommendation")))
    if "audit" in result: print("retrospective_audit=" + str(result["audit"]))
    print("caveat=historical model suggestion, not a causal or actionable prescription; feasibility does not guarantee actual effort or preserve size, quality, or other requirements; effort units are not invented")

def spot(rows, threshold=3.5, min_reference=3):
    """Leave-one-out robust screening of nonnegative Effort- in log1p space.

    Score is abs(x - median(reference)) / (1.4826 * MAD(reference)).  A zero
    MAD is deliberately undefined; no alternative scale is substituted.
    """
    if not math.isfinite(threshold) or threshold < 0: raise ValueError("threshold must be finite and nonnegative")
    if not isinstance(min_reference, int) or min_reference < 1: raise ValueError("min_reference must be a positive integer")
    usable = []
    records = []
    for p in rows:
        raw = p.get("values", {}).get("Effort-", "")
        y = _effort(raw)
        if y is None:
            reason = "missing or invalid/nonfinite Effort-"
            status = "invalid_or_missing"
        elif y < 0:
            reason = "negative Effort- is invalid for log1p transformation"
            status = "invalid_or_missing"
        else:
            usable.append((p, math.log1p(y)))
            status = None; reason = None
        records.append({"position": p.get("position"), "idX": p.get("values", {}).get("idX", ""),
                        "effort": y, "transformed": None if y is None or y < 0 else math.log1p(y),
                        "reference_count": None, "center": None, "scale": None, "score": None,
                        "threshold": threshold, "status": status, "reason": reason,
                        "descriptors": {c: _cell(p.get("values", {}).get(c, "")) for c in PREDICTORS}})
    for i, (p, x) in enumerate(usable):
        ref = [z for j, (_, z) in enumerate(usable) if j != i]
        r = records[p.get("position") and next(k for k, q in enumerate(records) if q["position"] == p.get("position")) or i]
        r["reference_count"] = len(ref)
        if len(ref) < min_reference:
            r["status"] = "undefined"; r["reason"] = "insufficient leave-one-out reference observations"; continue
        center = median(ref); mad = median([abs(z - center) for z in ref]); scale = 1.4826 * mad
        r["center"], r["scale"] = center, scale
        if scale == 0:
            r["status"] = "undefined"; r["reason"] = "zero MAD reference scale; score undefined by policy"; continue
        score = abs(x - center) / scale; r["score"] = score
        r["status"] = "flagged" if score >= threshold else "ordinary_nonflag"
        r["reason"] = "robust score meets threshold" if score >= threshold else "valid observation below threshold"
    counts = {k: sum(r["status"] == k for r in records) for k in ("flagged", "ordinary_nonflag", "undefined", "invalid_or_missing")}
    return {"records": records, "threshold": threshold, "min_reference": min_reference,
            "valid": counts["flagged"] + counts["ordinary_nonflag"] + counts["undefined"],
            "excluded": counts["invalid_or_missing"], "flagged": counts["flagged"],
            "ordinary_nonflag": counts["ordinary_nonflag"], "undefined": counts["undefined"],
            "invalid_or_missing": counts["invalid_or_missing"],
            "policy": "log1p nonnegative Effort-; leave-one-out median/MAD; scale=1.4826*MAD; zero MAD undefined; score=abs(x-center)/scale"}

def render_spot(path, threshold=3.5, min_reference=3):
    result = spot(load(path)["projects"], threshold, min_reference)
    print("NASA93 marginal historical Effort- anomaly screen")
    print("policy=" + result["policy"])
    print(f"threshold={threshold} min_reference={min_reference} valid={result['valid']} excluded={result['excluded']} flagged={result['flagged']} ordinary_nonflag={result['ordinary_nonflag']} undefined={result['undefined']}")
    print("convention=position is the physical CSV line (header is line 1); malformed rows are excluded by load and reported separately; idX is not assumed unique")
    print("limitation=large projects can legitimately require more effort; this marginal screen does not adjust for size or all descriptors; unusual results warrant investigation and do not prove measurement error; both low and high effort may be unusual")
    examples = [r for r in result["records"] if r["status"] in ("flagged", "undefined", "invalid_or_missing")]
    for r in examples[:12]: print("example=" + str(r))

def classify_difference(difference, tolerance):
    """Classify baseline-minus-estimator MAE; positive favors the estimator."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not math.isfinite(difference):
        return "undefined"
    if difference > tolerance: return "win"
    if difference < -tolerance: return "loss"
    return "tie"

def race(rows, seeds=range(42, 52), tolerance=1e-9):
    """Compare categorical KNN with a training-mean baseline on grouped splits."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    results = []
    for seed in seeds:
        try:
            train, test = split_groups(rows, seed)
        except ValueError as exc:
            results.append({"seed": seed, "status": "insufficient groups", "error": str(exc)})
            continue
        model = CategoricalKNN(train)
        train_invalid = sum(_effort(p["values"].get("Effort-", "")) is None for p in train)
        test_invalid = sum(_effort(p["values"].get("Effort-", "")) is None for p in test)
        scored = []
        for p in test:
            y = _effort(p["values"].get("Effort-", ""))
            if y is None or model.training_mean is None: continue
            pred = model.predict({c: p["values"].get(c, "") for c in PREDICTORS})
            scored.append({"position": p.get("position"), "idX": p["values"].get("idX", ""),
                           "actual": y, "prediction": pred,
                           "baseline_prediction": model.training_mean,
                           "model_error": abs(y - pred),
                           "baseline_error": abs(y - model.training_mean)})
        mae = statistics.fmean(x["model_error"] for x in scored) if scored else None
        bmae = statistics.fmean(x["baseline_error"] for x in scored) if scored else None
        diff = bmae - mae if mae is not None and bmae is not None else None
        results.append({"seed": seed, "status": "comparable" if scored else "no evaluable test targets",
                        "train_rows": len(train), "test_rows": len(test),
                        "train_groups": len(_groups(train)), "test_groups": len(_groups(test)),
                        "evaluated": len(scored), "excluded_train_targets": train_invalid,
                        "excluded_test_targets": test_invalid, "mae": mae,
                        "baseline_mae": bmae, "paired_difference": diff,
                        "classification": classify_difference(diff, tolerance) if diff is not None else "undefined",
                        "scored": scored})
    comparable = [x for x in results if x.get("paired_difference") is not None]
    diffs = [x["paired_difference"] for x in comparable]
    model_maes = [x["mae"] for x in comparable]; baseline_maes = [x["baseline_mae"] for x in comparable]
    counts = {label: sum(x["classification"] == label for x in comparable) for label in ("win", "tie", "loss")}
    return {"tolerance": tolerance, "seeds": results, "comparable_count": len(comparable),
            "mean_difference": statistics.fmean(diffs) if diffs else None,
            "median_difference": median(diffs),
            "estimator_mean_mae": statistics.fmean(model_maes) if model_maes else None,
            "baseline_mean_mae": statistics.fmean(baseline_maes) if baseline_maes else None,
            "wins": counts["win"], "ties": counts["tie"], "losses": counts["loss"]}

def render_race(path, tolerance=1e-9):
    result = race(load(path)["projects"], tolerance=tolerance)
    print("NASA93 grouped race: categorical estimator versus training-mean baseline")
    print(f"tie_tolerance={result['tolerance']} direction=baseline_MAE_minus_estimator_MAE; positive favors estimator")
    for x in result["seeds"]:
        print("seed_result=" + str({k: v for k, v in x.items() if k != "scored"}))
    print("summary=" + str({k: result[k] for k in ("comparable_count", "mean_difference", "median_difference", "estimator_mean_mae", "baseline_mean_mae", "wins", "ties", "losses")}))
    print("interpretation=repeated overlapping grouped splits of one historical dataset; not independent LLM trials or datasets; no significance, causation, or future superiority claim; effort units are not invented")

def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize NASA93 historical projects")
    ap.add_argument("dataset", help="CSV dataset path")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--flow", action="store_true")
    ap.add_argument("--blame", action="store_true")
    ap.add_argument("--justify", action="store_true")
    ap.add_argument("--choose", action="store_true")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--race", action="store_true")
    ap.add_argument("--tie-tolerance", type=float, default=1e-9)
    ap.add_argument("--effort-cap", type=float, default=100)
    args = ap.parse_args(argv)
    try: certify(pathlib.Path(args.dataset)) if args.certify else render_justify(pathlib.Path(args.dataset)) if args.justify else render_estimate(pathlib.Path(args.dataset)) if args.estimate else render_flow(pathlib.Path(args.dataset)) if args.flow else render_blame(load(pathlib.Path(args.dataset))) if args.blame else render_choose(pathlib.Path(args.dataset), args.effort_cap) if args.choose else render_fix(pathlib.Path(args.dataset), args.effort_cap) if args.fix else render_spot(pathlib.Path(args.dataset)) if args.spot else render_race(pathlib.Path(args.dataset), args.tie_tolerance) if args.race else render(load(pathlib.Path(args.dataset)))
    except (OSError, csv.Error) as exc: print(f"error: {exc}", file=sys.stderr); return 2
    return 0
if __name__ == "__main__": raise SystemExit(main())
