"""Summarize a CSV project-estimation dataset using only the standard library."""
from __future__ import annotations
import csv, math, sys
from collections import Counter
from pathlib import Path

MISSING = {"", "?", "na", "n/a", "null", "none", "missing"}
OUTCOMES = {"Kloc+": "maximize", "Effort-": "minimize", "Defects-": "minimize"}

def parse_cell(value):
    text = value.strip()
    if text.lower() in MISSING:
        return None
    try:
        number = float(text)
        if not math.isfinite(number):
            return None
        return number
    except ValueError:
        return text

def _stats(values):
    if not values:
        return {"count": 0, "mean": None, "sd": None, "min": None, "median": None, "max": None}
    ordered = sorted(values); n = len(values); mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    middle = (ordered[(n-1)//2] + ordered[n//2]) / 2
    return {"count": n, "mean": mean, "sd": math.sqrt(variance), "min": ordered[0], "median": middle, "max": ordered[-1]}

def summarize(path):
    """Read path and return a JSON-serializable summary; reject malformed outcomes."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, skipinitialspace=False)
        try: headers = [x.strip() for x in next(reader)]
        except StopIteration: return {"record_count": 0, "columns": [], "types": {}, "missing": {}, "numeric": {}, "categorical": {}, "roles": {"descriptors": [], "metadata": [], "outcomes": {}}}
        if not headers or any(not x for x in headers) or len(set(headers)) != len(headers):
            raise ValueError("header must contain unique non-empty names")
        rows = []
        for line, raw in enumerate(reader, 2):
            if not raw or all(not x.strip() for x in raw): continue
            if len(raw) != len(headers): raise ValueError(f"row {line} has {len(raw)} fields; expected {len(headers)}")
            parsed = [parse_cell(x) for x in raw]
            for name, value in zip(headers, parsed):
                if name in OUTCOMES and value is not None and not isinstance(value, float):
                    raise ValueError(f"invalid numeric outcome {name!r} on row {line}")
            rows.append(parsed)
    columns = {}
    for i, name in enumerate(headers):
        vals = [row[i] for row in rows if row[i] is not None]
        if not vals: kind = "unknown"
        elif all(isinstance(x, float) for x in vals): kind = "numeric"
        else: kind = "categorical"
        columns[name] = {"index": i, "type": kind, "missing": len(rows)-len(vals)}
    numeric, categorical = {}, {}
    for i, name in enumerate(headers):
        vals = [r[i] for r in rows if isinstance(r[i], float)]
        if columns[name]["type"] == "numeric": numeric[name] = _stats(vals)
        else:
            categorical[name] = dict(sorted(Counter(str(r[i]) for r in rows if r[i] is not None).items()))
    roles = {"descriptors": [], "metadata": [], "outcomes": {}}
    for name in headers:
        if name.endswith("X"): roles["metadata"].append(name)
        elif name in OUTCOMES: roles["outcomes"][name] = {"direction": OUTCOMES[name], "role": "outcome"}
        else: roles["descriptors"].append(name)
    missing = {name: columns[name]["missing"] for name in headers}
    return {"record_count": len(rows), "columns": headers, "types": {n: columns[n]["type"] for n in headers}, "missing": missing, "numeric": numeric, "categorical": categorical, "roles": roles}

def format_report(summary):
    lines = ["Historical project and delivery-outcome summary", f"Records: {summary['record_count']}", "Columns:"]
    for name in summary["columns"]:
        lines.append(f"  {name}: {summary['types'][name]}, missing={summary['missing'][name]}")
    lines += ["Numeric descriptive statistics (count, mean, sd, min, median, max):"]
    for name, stat in summary["numeric"].items():
        lines.append("  " + name + ": " + ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k,v in stat.items()))
    lines.append("Categorical frequencies:")
    for name, frequencies in summary["categorical"].items(): lines.append(f"  {name}: " + ", ".join(f"{k}={v}" for k,v in frequencies.items()))
    lines.append("Roles: descriptors=" + ", ".join(summary["roles"]["descriptors"]) + "; metadata=" + ", ".join(summary["roles"]["metadata"]))
    lines.append("Outcomes: " + ", ".join(f"{k} ({v['direction']})" for k,v in summary["roles"]["outcomes"].items()))
    lines.append("Interpretation: Kloc+ is delivered size, Effort- is effort, and Defects- is defects; X-suffix fields are excluded metadata. Numeric category codes are reported as numeric only when parsed as finite numbers, without assigning continuous meaning.")
    return "\n".join(lines) + "\n"

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1: raise SystemExit("usage: summary.py DATASET.csv")
    print(format_report(summarize(argv[0])), end="")

if __name__ == "__main__": main()
