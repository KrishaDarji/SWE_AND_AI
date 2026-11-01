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

def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize NASA93 historical projects")
    ap.add_argument("dataset", help="CSV dataset path")
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args(argv)
    try: render_estimate(pathlib.Path(args.dataset)) if args.estimate else render(load(pathlib.Path(args.dataset)))
    except (OSError, csv.Error) as exc: print(f"error: {exc}", file=sys.stderr); return 2
    return 0
if __name__ == "__main__": raise SystemExit(main())
