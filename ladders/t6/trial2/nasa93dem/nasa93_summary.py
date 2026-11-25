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

def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize or estimate NASA93 historical projects")
    parser.add_argument("dataset")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--certify", action="store_true")
    args = parser.parse_args(argv)
    if args.certify:
        _, rows, _, _ = load(args.dataset); ref=[parse_float(p.values.get("Effort-", ""))[0] for p in rows if norm_cell(p.values.get("tool", "")).lower()=="n" and parse_float(p.values.get("Effort-", ""))[1]=="valid"]
        threshold=0.10*(sum(ref)/len(ref)) if ref else None; print_certify(certify(rows, threshold))
    elif args.estimate: print_estimate(estimate(args.dataset))
    else: print_summary(summary(args.dataset))
    return 0

if __name__ == "__main__": sys.exit(main())
