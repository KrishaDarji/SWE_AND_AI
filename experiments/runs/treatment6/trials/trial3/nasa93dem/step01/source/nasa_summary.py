"""NASA93 historical-project loader and descriptive summary (stdlib only)."""
import argparse, csv, math, pathlib, statistics, sys

CATEGORICAL = {"prec","flex","resl","team","pmat","rely","data","cplx","ruse","docu","time","stor","pvol","acap","pcap","pcon","apex","plex","ltex","tool","site","sced"}
METADATA = {"idX","centerX","YearX","MonthsX"}
OUTCOMES = {"Kloc+":"maximize", "Effort-":"minimize", "Defects-":"minimize"}
MISSING = {"", "?", "NA", "N/A", "null", "NULL", "None"}

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
    args = ap.parse_args(argv)
    try: render(load(pathlib.Path(args.dataset)))
    except (OSError, csv.Error) as exc: print(f"error: {exc}", file=sys.stderr); return 2
    return 0
if __name__ == "__main__": raise SystemExit(main())
