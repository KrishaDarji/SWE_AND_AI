#!/usr/bin/env python3
"""Summarize model-configuration quality measurements."""
import csv, math, statistics, sys, random
from collections import Counter, defaultdict

CONFIG = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
OUTCOMES = {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"}
NUMERIC = {"N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"}
MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "missing"}

def finite_number(value):
    if value is None or value.strip().lower() in MISSING: return None
    try:
        x = float(value.strip())
        return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None

def median(values):
    return statistics.median(values) if values else None

def dispersion(values):
    return {"sample_stddev": statistics.stdev(values) if len(values) >= 2 else None,
            "sample_variance": statistics.variance(values) if len(values) >= 2 else None}

def fmt(x):
    return "missing" if x is None else (str(int(x)) if isinstance(x, float) and x.is_integer() else str(x))

def analyze(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f); rows = list(reader)
    if not rows: return {"path": path, "empty": True, "records": 0, "columns": [], "roles": {}, "malformed": [], "missing": {}, "invalid": {}, "outcomes": {}, "numeric": {}, "criterion": {}, "configurations": {"distinct": 0, "repeated": [], "exact_duplicate_rows": 0}}
    header = [x.strip() for x in rows[0]]
    roles = {c: ("outcome" if c in OUTCOMES else "configuration" if c in CONFIG else "other") for c in header}
    missing = Counter(); invalid = Counter(); records=[]; malformed=[]; exact=Counter()
    for pos, raw in enumerate(rows[1:], 2):
        if len(raw) != len(header): malformed.append(pos); continue
        row = {header[i]: raw[i].strip() for i in range(len(header))}; records.append((pos,row)); exact[tuple(raw)] += 1
        for c in header:
            if row[c].lower() in MISSING: missing[c] += 1
            elif c in NUMERIC or c in OUTCOMES:
                if finite_number(row[c]) is None: invalid[c] += 1
    numeric={}; criterion=Counter(); normalized=[]
    for c in CONFIG:
        vals=[]
        for _,r in records:
            if c not in r: continue
            if c == "criterion":
                if r[c].lower() not in MISSING: criterion[r[c].strip()] += 1
            else:
                x=finite_number(r[c]); vals += [] if x is None else [x]
        if c != "criterion": numeric[c] = stats(vals)
    for _,r in records:
        key=[]; valid=True
        for c in CONFIG:
            if c not in r: valid=False; break
            if c == "criterion": v=r[c].strip(); valid &= v.lower() not in MISSING; key.append(("cat",v))
            else:
                v=finite_number(r[c]); valid &= v is not None; key.append(("num",v))
        if valid: normalized.append(tuple(key))
    counts=Counter(normalized)
    return {"path":path,"empty":False,"records":len(records),"columns":header,"roles":roles,"malformed":malformed,"missing":dict(missing),"invalid":dict(invalid),"outcomes":{c:stats([finite_number(r[c]) for _,r in records if c in r and finite_number(r[c]) is not None]) | {"direction":d} for c,d in OUTCOMES.items()},"numeric":numeric,"criterion":dict(criterion),"configurations":{"distinct":len(counts),"repeated":[{"configuration":list(k),"count":n} for k,n in counts.items() if n>1],"exact_duplicate_rows":sum(n-1 for n in exact.values() if n>1)}}

def stats(vals):
    return {"finite_count":len(vals),"minimum":min(vals) if vals else None,"maximum":max(vals) if vals else None,"mean":statistics.mean(vals) if vals else None,"median":median(vals),**dispersion(vals)}

def _setting(row, name):
    if name == "criterion":
        v = row.get(name, "").strip()
        return None if v.lower() in MISSING else ("cat", v)
    v = finite_number(row.get(name, ""))
    return None if v is None else ("num", v)

def normalized_config(row):
    vals = tuple(_setting(row, c) for c in CONFIG)
    return vals if all(v is not None for v in vals) else None

def grouped_split(records, seed=42):
    groups = defaultdict(list)
    for rec in records:
        key = normalized_config(rec[1])
        if key is not None: groups[key].append(rec)
    if len(groups) < 2: raise ValueError("at least two complete configuration groups are required")
    keys = sorted(groups, key=repr); random.Random(seed).shuffle(keys)
    cut = max(1, min(len(keys)-1, int(round(len(keys)*0.8))))
    train_keys, test_keys = set(keys[:cut]), set(keys[cut:])
    train = [r for k in keys[:cut] for r in groups[k]]
    test = [r for k in keys[cut:] for r in groups[k]]
    return train, test, train_keys, test_keys

class RegressionTree:
    """Small CART-like median tree: numeric thresholds, categorical equality."""
    def __init__(self, max_depth=4, min_leaf=2): self.max_depth=max_depth; self.min_leaf=min_leaf; self.root=None; self.mean=None
    def fit(self, rows, target):
        data=[(r, finite_number(r.get(target,""))) for _,r in rows]; data=[x for x in data if x[1] is not None]
        if not data: raise ValueError("no finite training targets")
        self.mean=statistics.mean(y for _,y in data)
        self.root=(self.mean, None) if len(data)>1000 else self._grow(data,0)
        return self
    def _grow(self, data, depth):
        ys=[y for _,y in data]; node=(statistics.mean(ys), None)
        if depth>=self.max_depth or len(data)<2*self.min_leaf or len(set(ys))==1: return node
        best=None; base=sum((y-statistics.mean(ys))**2 for y in ys)
        for c in CONFIG:
            vals=[_setting(r,c) for r,_ in data]
            candidates=[]
            if c=="criterion": candidates=[v for v in sorted(set(vals),key=repr) if v is not None]
            else:
                nums=sorted(set(v[1] for v in vals if v is not None))
                if len(nums)>3:
                    nums=[nums[0], nums[len(nums)//2], nums[-1]]
                candidates=[(nums[i]+nums[i+1])/2 for i in range(len(nums)-1)]
            for cut in candidates:
                if c=="criterion": left=[x for x,v in zip(data,vals) if v==cut]; right=[x for x,v in zip(data,vals) if v!=cut]
                else: left=[x for x,v in zip(data,vals) if v is not None and v[1]<=cut]; right=[x for x,v in zip(data,vals) if v is None or v[1]>cut]
                if len(left)<self.min_leaf or len(right)<self.min_leaf: continue
                score=sum((y-statistics.mean([z for _,z in q]))**2 for q in (left,right) for _,y in q)
                if best is None or score<best[0]: best=(score,c,cut,left,right)
        if best is None or best[0]>=base: return node
        _,c,cut,left,right=best; return (node[0], (c,cut,self._grow(left,depth+1),self._grow(right,depth+1)))
    def predict_one(self,row):
        node=self.root
        while node and node[1]:
            c,cut,left,right=node[1]; v=_setting(row,c)
            if c=="criterion": node=left if v==cut else right
            else: node=left if v is not None and v[1]<=cut else right
        return node[0] if node else self.mean

def evaluate(path, seed=42):
    with open(path, newline="", encoding="utf-8") as f: raw=list(csv.DictReader(f))
    records=[(i+2,{k:(v or "").strip() for k,v in enumerate([])}) for i in []] # keep positions explicit below
    reader=[]
    with open(path, newline="", encoding="utf-8") as f:
        for pos,row in enumerate(csv.DictReader(f),2): reader.append((pos,{k:(v or "").strip() for k,v in row.items()}))
    train,test,train_keys,test_keys=grouped_split(reader,seed); result={"predictors":CONFIG,"outcome_exclusions":sorted(OUTCOMES),"train_rows":len(train),"test_rows":len(test),"train_groups":len(train_keys),"test_groups":len(test_keys),"targets":{}}
    for target in ("MRE-","ACC+"):
        model=RegressionTree(max_depth=1, min_leaf=16).fit(train,target); actual=[]; pred=[]
        for pos,row in test:
            y=finite_number(row.get(target,""));
            if y is not None: actual.append(y); pred.append(model.predict_one(row))
        mean=statistics.mean([finite_number(r.get(target,"")) for _,r in train if finite_number(r.get(target,"")) is not None])
        mae=sum(abs(a-p) for a,p in zip(actual,pred))/len(actual) if actual else None
        base=sum(abs(a-mean) for a in actual)/len(actual) if actual else None
        result["targets"][target]={"finite_training":sum(finite_number(r.get(target,"")) is not None for _,r in train),"finite_test":len(actual),"mae":mae,"baseline_mae":base,"training_mean":mean,"model":model}
    return result

def predict_configuration(model, config):
    if set(config) != set(CONFIG): raise ValueError("configuration must contain exactly the five setting fields")
    return {target:model["targets"][target]["model"].predict_one(config) for target in ("MRE-","ACC+")}

def render(a):
    if a.get("empty"): return "Empty input: zero records and no columns."
    lines=["Model-configuration quality summary",f"Records: {a['records']}",f"Columns: {', '.join(a['columns'])}",f"Roles: {a['roles']}",f"Malformed data rows (original CSV positions): {a['malformed'] or 'none'}",f"Missing tokens: {sorted(MISSING)}",f"Missing values: {a['missing'] or 'none'}",f"Invalid/nonfinite numeric values: {a['invalid'] or 'none'}","Configuration numeric summaries:"]
    for c,s in a["numeric"].items(): lines.append(f"  {c}: {s}")
    lines.append(f"criterion frequencies (categorical): {a['criterion']}")
    for c,s in a["outcomes"].items(): lines.append(f"{c} ({s['direction']}): {s}")
    lines += [f"Normalized distinct complete configurations: {a['configurations']['distinct']}",f"Repeated normalized configurations: {a['configurations']['repeated']}",f"Exact duplicate data rows beyond first: {a['configurations']['exact_duplicate_rows']}","Normalization: numeric equivalents use float values; categorical whitespace is stripped; missing/invalid settings are excluded from configuration keys.","Dispersion: sample standard deviation and sample variance; both are missing when fewer than two finite observations exist.","Rows retain original CSV positions (header is position 1); malformed rows are excluded. Missing outcomes remain excluded from that outcome's finite summary.","These are model-configuration quality measurements, not an issue-closure timeline."]
    return "\n".join(lines)

def main(argv):
    if len(argv) != 2: print("usage: summarize_runs.py DATASET.csv", file=sys.stderr); return 2
    print(render(analyze(argv[1]))); return 0
if __name__ == "__main__": raise SystemExit(main(sys.argv))
