"""NASA93 historical-project loader and descriptive summary (stdlib only)."""
import argparse, csv, math, pathlib, statistics, sys
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

def render_estimate(path):
    result = evaluate(load(path)["projects"])
    print(f"train_rows={len(result['train'])} test_rows={len(result['test'])} train_groups={result['train_groups']} test_groups={result['test_groups']}")
    print(f"excluded_targets={result['excluded_targets']} heldout_mae={result['mae']} training_mean_baseline_mae={result['baseline_mae']}")

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

def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize NASA93 historical projects")
    ap.add_argument("dataset", help="CSV dataset path")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--certify", action="store_true")
    args = ap.parse_args(argv)
    try: certify(pathlib.Path(args.dataset)) if args.certify else render_estimate(pathlib.Path(args.dataset)) if args.estimate else render(load(pathlib.Path(args.dataset)))
    except (OSError, csv.Error) as exc: print(f"error: {exc}", file=sys.stderr); return 2
    return 0
if __name__ == "__main__": raise SystemExit(main())
