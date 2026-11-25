#!/usr/bin/env python3
"""Configuration analysis and held-out prediction using only the stdlib."""
import csv, math, random, statistics
CONFIGS={"N_estimators":"numeric","criterion":"categorical","Min_sample_leaves":"numeric","Min_impurity_decrease":"numeric","Max_depth":"numeric"}
OUTCOMES={"MRE-":"minimize","ACC+":"maximize","PRED40+":"maximize"}; MISSING={"","?","NA","N/A","null","None","nan","NaN"}
def _number(v):
    if v is None:return None
    v=str(v)
    if v.strip() in MISSING:return None
    try:
        x=float(v.strip()); return x if math.isfinite(x) else None
    except (TypeError,ValueError): return None
def _cat(v): return None if v is None or str(v).strip() in MISSING else str(v).strip()
def _median(x): return statistics.median(x) if x else None
def _summary(x):
    if not x:return {"finite_count":0,"minimum":None,"maximum":None,"mean":None,"median":None,"sample_sd":None,"iqr":None,"convention":"insufficient sample: null"}
    q=statistics.quantiles(x,n=4,method="inclusive") if len(x)>1 else []
    return {"finite_count":len(x),"minimum":min(x),"maximum":max(x),"mean":statistics.fmean(x),"median":_median(x),"sample_sd":statistics.stdev(x) if len(x)>1 else None,"iqr":q[2]-q[0] if q else None,"convention":"sample standard deviation (n-1); inclusive quartiles; null when n<2"}
def _read(path):
    with open(path,newline="",encoding="utf-8-sig") as f:
        rd=csv.reader(f); raw=next(rd,None)
        if raw is None:return [],[],[]
        h=[x.strip() for x in raw]; rows=[]; bad=[]
        for pos,c in enumerate(rd,2):
            if not c or all(not x.strip() for x in c):continue
            if len(c)!=len(h):bad.append(pos);continue
            rows.append((pos,dict(zip(h,c))))
    return h,rows,bad
def _key(r): return tuple(_cat(r.get(k)) if t=="categorical" else _number(r.get(k)) for k,t in CONFIGS.items())
def analyze(path):
    h,rows,bad=_read(path); z=empty_result("empty input" if not h else ("header-only input" if not rows else None)); z.update({"record_count":len(rows),"column_names":h,"malformed_row_positions":bad,"roles":{**{k:"configuration ("+v+")" for k,v in CONFIGS.items()},**{k:"outcome ("+v+")" for k,v in OUTCOMES.items()}},"missing_values":{k:sum(r.get(k,"").strip() in MISSING for _,r in rows) for k in h},"invalid_nonfinite_numeric":{k:sum(r.get(k,"").strip() not in MISSING and _number(r.get(k)) is None for _,r in rows) for k in list(CONFIGS)+list(OUTCOMES) if k in h},"criterion_frequencies":{}})
    for _,r in rows:
        c=_cat(r.get("criterion"));
        if c is not None:z["criterion_frequencies"][c]=z["criterion_frequencies"].get(c,0)+1
    z["numeric_settings"]={k:_summary([_number(r.get(k)) for _,r in rows if _number(r.get(k)) is not None]) for k,t in CONFIGS.items() if t=="numeric" and k in h}; z["outcomes"]={k:{"direction":d,**_summary([_number(r.get(k)) for _,r in rows if _number(r.get(k)) is not None])} for k,d in OUTCOMES.items() if k in h}
    g={}; e={}
    for p,r in rows:g.setdefault(_key(r),[]).append(p); e.setdefault(tuple(r.get(k,"") for k in h),[]).append(p)
    z["normalized_configurations"]=[{"settings":dict(zip(CONFIGS,k)),"row_positions":v} for k,v in g.items()]; z["repeated_configurations"]=[x for x in z["normalized_configurations"] if len(x["row_positions"])>1]; z["exact_duplicate_row_groups"]=[v for v in e.values() if len(v)>1]; z["normalization"]="Trim categorical whitespace; numeric equivalents use float; missing/invalid settings become null; rows remain position-addressable."; return z
class _Tree:
    def __init__(self,rows,target,depth=2,leaf=500):
        self.rows=rows; self.target=target; self.depth=depth; self.leaf=leaf
        self.root=("leaf",self.mean(list(range(len(rows))))) if len(rows)>2000 else self.grow(list(range(len(rows))),0)
    def y(self,i):return _number(self.rows[i].get(self.target))
    def mean(self,a):return statistics.fmean([self.y(i) for i in a])
    def grow(self,a,d):
        node=("leaf",self.mean(a))
        if d>=self.depth or len(a)<2*self.leaf:return node
        best=(sum((self.y(i)-self.mean(a))**2 for i in a),None)
        for f,t in CONFIGS.items():
            v=[(_number(self.rows[i].get(f)) if t=="numeric" else _cat(self.rows[i].get(f)),i) for i in a]; u=sorted({x for x,i in v if x is not None});
            if t=="numeric":
                cuts=[(u[int(len(u)*j/4)]+u[min(len(u)-1,int(len(u)*j/4)+1)])/2 for j in range(1,4) if u and int(len(u)*j/4)<len(u)-1]
            else: cuts=u
            for c in cuts:
                l=[i for x,i in v if x is not None and (x<c if t=="numeric" else x==c)]; r=[i for i in a if i not in l]
                score=0
                if len(l)>=self.leaf and len(r)>=self.leaf:
                    for q in (l,r):
                        mq=self.mean(q); score += sum((self.y(i)-mq)**2 for i in q)
                else: score=best[0]
                if score<best[0]:best=(score,(f,t,c,l,r))
        if best[1] is None:return node
        f,t,c,l,r=best[1]; return ("split",f,t,c,self.grow(l,d+1),self.grow(r,d+1))
    def predict(self,row):
        n=self.root
        while n[0]=="split":
            _,f,t,c,l,r=n; v=_number(row.get(f)) if t=="numeric" else _cat(row.get(f)); n=l if v is not None and (v<c if t=="numeric" else v==c) else r
        return n[1]
def _mae(a,p):return statistics.fmean(abs(x-y) for x,y in zip(a,p)) if a else None
def predict_dataset(path,seed=42):
    h,rows,bad=_read(path); g={}
    for p,r in rows:g.setdefault(_key(r),[]).append((p,r))
    if len(g)<2:raise ValueError("at least two normalized configuration groups are required")
    keys=sorted(g,key=repr); random.Random(seed).shuffle(keys); cut=max(1,min(len(keys)-1,round(len(keys)*.8))); train=[x for k in keys[:cut] for x in g[k]]; test=[x for k in keys[cut:] for x in g[k]]; out={"predictors":list(CONFIGS),"outcome_exclusions":list(OUTCOMES),"train_rows":len(train),"test_rows":len(test),"train_groups":cut,"test_groups":len(keys)-cut,"seed":seed,"targets":{},"models":"held-out models fitted only on training rows; no all-history model is fitted"}
    for target in ("MRE-","ACC+"):
        usable=[(p,r) for p,r in train if _number(r.get(target)) is not None]; actual=[(p,r,_number(r.get(target))) for p,r in test if _number(r.get(target)) is not None]; m=_Tree([r for _,r in usable],target) if len(usable)>=2 else None; mean=statistics.fmean([_number(r.get(target)) for _,r in usable]) if usable else None; pred=[m.predict(r) for _,r,_ in actual] if m else []; a=[v for _,_,v in actual]
        out["targets"][target]={"finite_training_targets":len(usable),"evaluated_test_targets":len(a),"excluded_training_targets":len(train)-len(usable),"excluded_test_targets":len(test)-len(a),"held_out_mae":_mae(a,pred),"baseline_training_mean":mean,"baseline_mae":_mae(a,[mean]*len(a)) if mean is not None else None,"evaluations":[{"row_position":p,"actual":v,"prediction":q} for (p,r,v),q in zip(actual,pred)]}
    return out
def predict_configuration(configuration,model):
    if set(configuration)!=set(CONFIGS):raise ValueError("configuration must contain exactly the five setting fields")
    return model.predict(configuration)
def empty_result(reason):return {"record_count":0,"column_names":[],"roles":{},"missing_values":{},"invalid_nonfinite_numeric":{},"criterion_frequencies":{},"numeric_settings":{},"outcomes":{},"normalized_configurations":[],"repeated_configurations":[],"exact_duplicate_row_groups":[],"malformed_row_positions":[],"note":reason}

# Flow is deliberately separate from the batch summaries above.  It follows the
# same missing-value convention, but never materializes the future input.
def validation_rows(path):
    """Yield (CSV position, row) lazily; blank and malformed records are skipped."""
    def rows():
        with open(path,newline="",encoding="utf-8-sig") as f:
            rd=csv.reader(f); header=next(rd,None)
            if header is None:return
            for position,cells in enumerate(rd,2):
                if not cells or all(not cell.strip() for cell in cells):continue
                if len(cells)!=len(header):continue
                yield position,dict(zip((x.strip() for x in header),cells))
    return rows()

class _Running:
    def __init__(self): self.n=0; self.mean=0.0; self.m2=0.0
    def add(self,x):
        self.n+=1; delta=x-self.mean; self.mean+=delta/self.n; self.m2+=delta*(x-self.mean)
    def result(self):
        return {"finite_count":self.n,"mean":self.mean if self.n else None,"sample_variance":self.m2/(self.n-1) if self.n>1 else None}

def _best(current,value,position,row,direction):
    if value is None:return current
    if current is None or (value<current["value"] if direction=="minimize" else value>current["value"]):
        settings={k:(_cat(row.get(k)) if t=="categorical" else _number(row.get(k))) for k,t in CONFIGS.items()}
        return {"value":value,"position":position,"configuration":settings}
    return current

class FlowEvaluator:
    """One-pass prefix evaluator; each snapshot is a detached value object."""
    def __init__(self):
        self.total_rows_seen=0; self._stats={k:_Running() for k in OUTCOMES}; self._best={k:None for k in OUTCOMES}
    def update(self,position,row):
        self.total_rows_seen+=1
        for outcome,direction in OUTCOMES.items():
            value=_number(row.get(outcome))
            if value is not None:self._stats[outcome].add(value)
            self._best[outcome]=_best(self._best[outcome],value,position,row,direction)
        return self.snapshot()
    def snapshot(self):
        stats={k:self._stats[k].result() for k in OUTCOMES}
        return {"total_rows_seen":self.total_rows_seen,"outcomes":stats,"best":{k:(None if v is None else {"value":v["value"],"position":v["position"],"configuration":dict(v["configuration"])}) for k,v in self._best.items()},"conventions":{"position":"physical CSV row, header is 1","blank_rows":"ignored and do not increment total_rows_seen","malformed_rows":"ignored and do not increment total_rows_seen","missing_configuration":"retained as null; invalid numeric settings are null","missing_or_nonfinite_outcome":"excluded only from that outcome's count and statistics","ties":"earliest row retained"}}

def flow(rows):
    """Consume an iterable of (position, mapping), returning independent snapshots."""
    evaluator=FlowEvaluator(); snapshots=[]
    for position,row in rows:snapshots.append(evaluator.update(position,row))
    return snapshots

def flow_dataset(path):
    """Evaluate the CSV lazily and return only the final prefix snapshot."""
    evaluator=FlowEvaluator()
    for position,row in validation_rows(path):evaluator.update(position,row)
    return evaluator.snapshot()

def blame(path):
    """Rank marginal setting associations with lower-is-better MRE-.

    Numeric settings use up to four equal-frequency bins, with boundaries
    computed from that field's eligible finite setting values before outcomes
    are inspected.  Groups use interval notation [lo, hi] (the final group
    includes its upper endpoint); categorical values are exact strings.
    """
    _, rows, _ = _read(path)
    result = {"outcome": "MRE-", "score": "eta_squared = between-group SS / total SS",
              "lower_mre_is_better": True, "fields": list(CONFIGS), "ranking": [],
              "conventions": {"numeric_groups": "up to four equal-frequency bins; ties may reduce groups",
                               "intervals": "left-closed, right-open except final group",
                               "ties": "descending score, then field declaration order",
                               "association": "marginal historical association; not causation or significance"}}
    for field, kind in CONFIGS.items():
        eligible = []
        for pos, row in rows:
            y = _number(row.get("MRE-")); x = _cat(row.get(field)) if kind == "categorical" else _number(row.get(field))
            if x is not None and y is not None: eligible.append((pos, x, y))
        groups = []
        boundaries = []
        if kind == "categorical":
            for value in sorted({x for _, x, _ in eligible}, key=repr):
                groups.append((str(value), [y for _, x, y in eligible if x == value]))
        else:
            values = sorted(x for _, x, _ in eligible)
            if values:
                cuts = sorted({(values[j-1] + values[j]) / 2 for i in (1, 2, 3)
                               for j in [(len(values) * i) // 4]
                               if 0 < j < len(values) and values[j-1] != values[j]})
                boundaries = cuts
                for i in range(len(cuts) + 1):
                    lo = values[0] if i == 0 else cuts[i-1]
                    hi = values[-1] if i == len(cuts) else cuts[i]
                    ys = [y for _, x, y in eligible if (x >= lo if i == 0 else x > lo) and (x <= hi)]
                    if ys: groups.append(("[%s, %s]%s" % (lo, hi, "" if i == len(cuts) else " (right boundary shared)"), ys))
        n = len(eligible); mean = statistics.fmean(y for _, _, y in eligible) if n else None
        total = sum((y - mean) ** 2 for _, _, y in eligible) if n else 0.0
        between = sum(len(ys) * (statistics.fmean(ys) - mean) ** 2 for _, ys in groups) if n else 0.0
        score = between / total if n >= 2 and total > 0 and len(groups) >= 2 else None
        detail = [{"group": label, "count": len(ys), "mean_mre": statistics.fmean(ys)} for label, ys in groups]
        result["ranking"].append({"field": field, "kind": kind, "eligible_count": n,
            "excluded_count": len(rows) - n, "group_count": len(groups), "boundaries": boundaries,
            "reference_mean": mean, "total_ss": total, "between_group_ss": between,
            "eta_squared": score, "groups": detail,
            "status": "ok" if score is not None else "undefined: empty, insufficient, constant setting/outcome, or fewer than two groups"})
    result["ranking"].sort(key=lambda x: (x["eta_squared"] is None, -(x["eta_squared"] or 0), list(CONFIGS).index(x["field"])))
    return result

def _finite_outcome(row, name):
    return _number(row.get(name))

def _justify_rows(rows):
    eligible=[]
    for position,row in rows:
        settings={k:(_cat(row.get(k)) if t=="categorical" else _number(row.get(k))) for k,t in CONFIGS.items()}
        outcomes={k:_finite_outcome(row,k) for k in OUTCOMES}
        if all(v is not None for v in settings.values()) and all(v is not None for v in outcomes.values()):
            eligible.append((position,settings,outcomes,row))
    return eligible

def justify(path):
    """Select the best observed complete row and compare it with independent medians."""
    header,rows,bad=_read(path)
    result={"status":"ok","dataset_rows":len(rows),"malformed_row_positions":bad,
            "position_convention":"original physical CSV row; header is position 1; blank rows are ignored and malformed rows are excluded without renumbering",
            "selection_rule":["lowest finite MRE-","highest finite ACC+ for ties","highest finite PRED40+ for remaining ties","earliest original CSV position for remaining ties"],
            "eligibility":"a row needs finite values for all five settings and all three outcomes; criterion is categorical and other settings are numeric",
            "reference_convention":"each outcome median uses all finite historical values for that outcome independently; count is reported",
            "direction":{"MRE-":"minimize","ACC+":"maximize","PRED40+":"maximize"},"recommendation":None,"references":{},"comparisons":{},"association_context":None}
    if not header:
        result.update(status="empty_data",message="No header or data rows are available."); return result
    eligible=_justify_rows(rows)
    if not eligible:
        result.update(status="no_eligible_recommendation",message="No row has a complete finite configuration and all three finite outcomes."); return result
    chosen=min(eligible,key=lambda x:(x[2]["MRE-"],-x[2]["ACC+"],-x[2]["PRED40+"],x[0]))
    result["recommendation"]={"original_position":chosen[0],"configuration":chosen[1],"outcomes":chosen[2]}
    for outcome,direction in OUTCOMES.items():
        vals=[_number(row.get(outcome)) for _,row in rows if _number(row.get(outcome)) is not None]
        med=_median(vals); value=chosen[2][outcome]
        result["references"][outcome]={"median":med,"finite_count":len(vals)}
        better=(value<med if direction=="minimize" else value>med) if med is not None else None
        result["comparisons"][outcome]={"recommendation":value,"reference_median":med,"difference":value-med if med is not None else None,"direction":direction,"better_than_median":better}
    assoc=blame(path)
    result["association_context"]={"label":"global marginal historical associations, not row-level contributions or causal effects","outcome":assoc["outcome"],"grouping_conventions":assoc["conventions"],"ranking":assoc["ranking"]}
    result["interpretation"]="This is a retrospective recommendation from observed complete rows, not a prediction for an unevaluated configuration. Differences describe the selected row relative to each independently populated historical median; secondary outcomes can trade off against the primary MRE- rule. Global association scores provide context only and do not guarantee benefit."
    return result

def choose(path, cap=100):
    """Retrospectively select the best observed complete row within an estimator cap."""
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 0:
        raise ValueError("cap must be a finite nonnegative number")
    header, rows, bad = _read(path)
    result = {"status": "ok", "cap": cap, "raw_count": len(rows) + len(bad),
              "parsed_count": len(rows), "malformed_count": len(bad),
              "invalid_count": 0, "over_budget_count": 0, "feasible_count": 0,
              "malformed_row_positions": bad, "recommendation": None,
              "selection_rule": ["lowest MRE-", "highest ACC+ for ties",
                                 "highest PRED40+ for remaining ties",
                                 "earliest original CSV position for remaining ties"],
              "eligibility": "all five settings and all three outcomes finite; criterion nonmissing categorical; N_estimators nonnegative and <= cap",
              "interpretation": "Retrospective selection using observed outcomes; other settings may also affect computation. No optimality beyond eligible historical rows or guaranteed unseen performance is claimed.",
              "position_convention": "original physical CSV row; header is position 1; blank rows ignored; malformed rows excluded without renumbering"}
    if not header:
        result["status"] = "empty_data"; return result
    feasible = []
    for position, row in rows:
        settings = {k: (_cat(row.get(k)) if t == "categorical" else _number(row.get(k))) for k, t in CONFIGS.items()}
        outcomes = {k: _number(row.get(k)) for k in OUTCOMES}
        complete = all(v is not None for v in settings.values()) and all(v is not None for v in outcomes.values())
        if not complete:
            result["invalid_count"] += 1; continue
        if settings["N_estimators"] < 0:
            result["invalid_count"] += 1; continue
        if settings["N_estimators"] > cap:
            result["over_budget_count"] += 1; continue
        feasible.append((position, settings, outcomes))
    result["feasible_count"] = len(feasible)
    if not feasible:
        result["status"] = "no_feasible_candidate"; return result
    chosen = min(feasible, key=lambda x: (x[2]["MRE-"], -x[2]["ACC+"], -x[2]["PRED40+"], x[0]))
    result["recommendation"] = {"original_position": chosen[0], "configuration": chosen[1],
                                "outcomes": chosen[2], "selected_proxy_value": chosen[1]["N_estimators"],
                                "feasibility": "feasible: finite complete row and 0 <= N_estimators <= cap"}
    return result
