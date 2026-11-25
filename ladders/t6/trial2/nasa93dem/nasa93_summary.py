#!/usr/bin/env python3
"""Standard-library NASA93 loader and descriptive summary."""
import argparse, csv, math, sys
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

def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize NASA93 historical projects")
    parser.add_argument("dataset")
    args = parser.parse_args(argv)
    print_summary(summary(args.dataset)); return 0

if __name__ == "__main__": sys.exit(main())
